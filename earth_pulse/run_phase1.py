"""Phase 1 pipeline. Usage:
  python run_phase1.py --probe                 # print property keys of a few real results
  python run_phase1.py --probe-cmr             # list NISAR collections in CMR (+ Harmony status)
  python run_phase1.py --refresh               # live ASF search -> manifests/inventory/stacks
  python run_phase1.py --from-cache            # rebuild outputs from the last live search
  python run_phase1.py --refresh --download 1  # also download+inspect+read the best stack's 1st product
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone

from modules import config as C
from modules.data_validation import build_quality_report, validate_metadata
from modules.nasa_api import ArchiveAccessError, raw_search
from modules.nisar_search import search_archive
from modules.observation_ranking import build_stacks, explain_selection
from modules.product_metadata import dedupe, is_qa_or_auxiliary, normalize_record

CACHE = C.META_DIR / "search_cache.json"
STATE = C.META_DIR / "search_state.json"


def _utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def build_outputs(records, search_report, origin, refreshed_at):
    normalized = [normalize_record(r) for r in records]
    fixtures = [o["product_id"] for o in normalized if o["is_test_fixture"]]
    if fixtures:
        raise RuntimeError(f"Refusing to proceed: test-fixture records in real pipeline: {fixtures[:3]}")
    for o in normalized:
        o["qa_or_auxiliary"] = is_qa_or_auxiliary(o)
    science = [o for o in normalized if not o["qa_or_auxiliary"]]
    unique, dd = dedupe(science)
    stacks, stack_info = build_stacks(unique)
    qa = build_quality_report(unique, dd, search_report)
    qa["metadata_validation"] = [validate_metadata(o) for o in unique]
    qa["excluded_qa_or_auxiliary"] = len(normalized) - len(science)
    qa["stack_info"] = stack_info
    reason = explain_selection(stacks)
    C.ensure_dirs()
    m = C.META_DIR
    write_csv(m / "nisar_search_manifest.csv", normalized)
    (m / "nisar_search_manifest.json").write_text(json.dumps(
        {"origin": origin, "last_successful_refresh_utc": refreshed_at,
         "search": {k: v for k, v in search_report.items() if k != "windows"},
         "records": normalized}, indent=2, default=str))
    write_csv(m / "observation_inventory.csv", unique)
    write_csv(m / "candidate_stacks.csv", stacks)
    (m / "data_quality_report.json").write_text(json.dumps(qa, indent=2, default=str))
    summary = [
        f"EARTH PULSE - Phase 1 search summary",
        f"Origin of records: {origin}   Last successful live refresh (UTC): {refreshed_at}",
        f"Raw records returned: {search_report['raw_record_count']}  Windows queried: {search_report['n_windows']}",
        f"Search complete (no unresolved truncation): {search_report['complete']}",
        f"QA/auxiliary excluded: {qa['excluded_qa_or_auxiliary']}  Duplicates removed: {dd['duplicates_removed']}",
        f"Unique science observations: {len(unique)}  Complete metadata: {qa['n_complete_metadata']}",
        f"Stacks: {len(stacks)}  Eligible: {sum(s['eligible'] for s in stacks)}",
        f"Selection: {reason}"]
    (m / "search_summary.txt").write_text("\n".join(summary) + "\n")
    STATE.write_text(json.dumps({"origin": origin, "last_successful_refresh_utc": refreshed_at,
                                 "n_observations": len(unique), "n_stacks": len(stacks),
                                 "search_complete": search_report["complete"],
                                 "retrieval_status": "metadata only (no products downloaded)",
                                 "selection_reason": reason}, indent=2))
    print("\n".join(summary))
    return unique, stacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-cmr", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--start", default=C.DEFAULT_SEARCH_START)
    ap.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--chunk-days", type=int, default=None)
    ap.add_argument("--aoi", action="append", default=[], help="'lon,lat' or 'minlon,minlat,maxlon,maxlat' (repeatable)")
    ap.add_argument("--grid-step", type=float, default=None, help="global SAMPLE: query one point per N-degree cell")
    ap.add_argument("--all-archive", action="store_true", help="list the ENTIRE archive (huge; not recommended)")
    ap.add_argument("--cap", type=int, default=C.SEARCH_CAP, help="max results per request")
    ap.add_argument("--max-calls", type=int, default=60, help="hard limit on archive requests")
    ap.add_argument("--download", type=int, default=0, help="download first N products of best stack")
    a = ap.parse_args()
    C.ensure_dirs()
    try:
        if a.probe_cmr:
            from modules.harmony_probe import probe_cmr
            print(json.dumps(probe_cmr(), indent=2)); return 0
        if a.probe:
            recs, complete = raw_search(_utc(a.start), _utc(a.end), 5)
            print(f"returned={len(recs)} searchComplete={complete}")
            for r in recs[:3]:
                print(json.dumps(r["properties"], indent=2, default=str))
            return 0
        if a.from_cache:
            if not CACHE.exists():
                print("No cached live search exists. Run --refresh first.", file=sys.stderr); return 2
            c = json.loads(CACHE.read_text())
            build_outputs(c["records"], c["search_report"], "LOCAL_CACHE_OF_LIVE_SEARCH", c["refreshed_at"])
            return 0
        if a.refresh:
            from modules.aoi import grid_points, parse_aoi, point_wkt
            targets = [parse_aoi(t) for t in a.aoi]
            if a.grid_step:
                targets += [(f"grid({la:.0f},{lo:.0f})", point_wkt(lo, la)) for la, lo in grid_points(a.grid_step)]
            if not targets and not a.all_archive:
                print("Choose where to search. The archive holds hundreds+ of products PER DAY, so listing all of it "
                      "exhausts memory.\n  --aoi lon,lat            one place (repeatable)\n"
                      "  --aoi minlon,minlat,maxlon,maxlat   a box\n  --grid-step 30           global SAMPLE, one point per 30 deg cell\n"
                      "  --all-archive            everything (not recommended)", file=sys.stderr)
                return 5
            if not targets:
                targets = [("global", None)]
            span = max(1, (_utc(a.end) - _utc(a.start)).days + 1)
            chunk = a.chunk_days or (30 if targets[0][1] is None else span)
            all_recs, reports, aoi_rows = [], [], []
            for n, (label, wkt) in enumerate(targets, 1):
                def show(w, label=label, n=n):
                    print(f"[{n}/{len(targets)} {label}] call {w['call']}/{w['max_calls']} "
                          f"{w['start'][:10]} -> {w['end'][:10]}: {w['returned']} records"
                          f"{' (full page: splitting)' if w['split'] else ''}", flush=True)
                recs, rep = search_archive(lambda s, e, cap, wkt=wkt: raw_search(s, e, cap, wkt=wkt),
                                           _utc(a.start), _utc(a.end), chunk, a.cap, a.max_calls, show)
                all_recs += recs; reports.append(rep)
                aoi_rows.append({"aoi": label, "wkt": wkt, "records": len(recs), "complete": rep["complete"]})
            report = {"n_windows": sum(r["n_windows"] for r in reports), "complete": all(r["complete"] for r in reports),
                      "unresolved_truncated_windows": [w for r in reports for w in r["unresolved_truncated_windows"]],
                      "raw_record_count": len(all_recs), "calls_used": sum(r["calls_used"] for r in reports),
                      "budget_exhausted": any(r["budget_exhausted"] for r in reports),
                      "search_mode": "aoi/grid sample" if targets[0][1] else "entire archive",
                      "aois": aoi_rows, "windows": []}
            if report["budget_exhausted"]:
                print(f"WARNING: a search hit its {a.max_calls}-request limit; results are INCOMPLETE.", flush=True)
            records = all_recs
            now = datetime.now(timezone.utc).isoformat()
            with open(CACHE, "w") as fh:
                json.dump({"records": records, "search_report": report, "refreshed_at": now}, fh, default=str)
            unique, stacks = build_outputs(records, report, "LIVE_API", now)
            if a.download and stacks:
                from modules.data_extraction import download_product, find_gcov_layers, read_layer
                from modules.earthdata_auth import build_asf_session
                sess = build_asf_session()
                ids = stacks[0]["product_ids"].split(";")[: a.download]
                for o in [u for u in unique if u["product_id"] in ids]:
                    path, status = download_product(o, sess)
                    print(o["product_id"], status)
                    if path:
                        layers = find_gcov_layers(path)
                        print(json.dumps(layers, indent=2))
                        diag = [l for l in layers if l["diagonal"]]
                        if diag:
                            q = read_layer(path, diag[0]["path"])["quality"]
                            print("quality:", json.dumps(q))
            return 0
        ap.print_help(); return 1
    except ArchiveAccessError as e:
        msg = f"ARCHIVE ACCESS FAILURE (nothing was fabricated): {e}"
        (C.META_DIR / "search_summary.txt").write_text(msg + "\n")
        print(msg, file=sys.stderr); return 3
    except Exception as e:  # auth errors etc., already redacted at source
        print(f"{type(e).__name__}: {e}", file=sys.stderr); return 4


if __name__ == "__main__":
    sys.exit(main())
