"""Normalise raw ASF records into observation dicts. Nothing is invented: any field the API
does not provide stays None and is listed in `missing_fields`. Filename-derived values are
only a fallback and are labelled in `field_sources` (verify against the NISAR Data User Guide)."""
import json
import re
from datetime import datetime, timezone

from .config import MATURITY_TOKENS

REQUIRED_FIELDS = ("product_id", "start_time", "product_type", "track", "frame",
                   "direction", "polarization", "maturity", "release", "footprint_bbox")
TEST_FIXTURE_PREFIX = "TESTFIXTURE_"   # used only by unit tests; blocked from real pipeline
_TS = re.compile(r"\d{8}T\d{6}")
_CRID = re.compile(r"^[A-Z]\d{5}$")


def _first(props, *keys):
    for k in keys:
        v = props.get(k)
        if v is not None and v != "" and v != "NA":
            return v
    return None


def parse_time(value):
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_filename(name):
    """Best-effort token parse of a NISAR file name. Returns only what can be matched."""
    out = {}
    stem = re.sub(r"\.h5$", "", name or "", flags=re.I)
    toks = stem.split("_")
    if len(toks) > 7 and toks[0] == "NISAR":
        if toks[4].isdigit(): out["cycle"] = int(toks[4])
        if toks[5].isdigit(): out["track"] = int(toks[5])
        if toks[6] in ("A", "D"): out["direction"] = {"A": "ASCENDING", "D": "DESCENDING"}[toks[6]]
        if toks[7].isdigit(): out["frame"] = int(toks[7])
        out["product_type"] = toks[3]
    ts = _TS.findall(stem)
    if ts:
        out["start_time"] = datetime.strptime(ts[0], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()
    crid = [t for t in toks if _CRID.match(t)]
    if crid:
        out["release"] = crid[0]
    return out


def _coords(obj):
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
            yield (float(obj[0]), float(obj[1]))
        else:
            for o in obj:
                yield from _coords(o)


def footprint_bbox(geometry):
    """(min_lon, min_lat, max_lon, max_lat) or None. Antimeridian crossings are flagged
    by the caller via lon span > 180 and not handled geometrically (documented limitation)."""
    if not geometry or "coordinates" not in geometry:
        return None
    pts = list(_coords(geometry["coordinates"]))
    if not pts:
        return None
    xs, ys = zip(*pts)
    return (min(xs), min(ys), max(xs), max(ys))


def detect_maturity(props):
    """Look for a maturity token verbatim in API metadata values. None if not found."""
    for k, v in props.items():
        if isinstance(v, str):
            for tok in MATURITY_TOKENS:
                if tok in v.upper():
                    return tok, k
    return None, None


def normalize_record(record):
    props = record.get("properties") or {}
    name = _first(props, "fileName", "sceneName", "fileID")
    fn = parse_filename(name or "")
    src, obs = {}, {}

    def pick(field, api_keys, fallback=None):
        v = _first(props, *api_keys)
        if v is not None:
            src[field] = "asf_api"
            return v
        if fallback is not None:
            src[field] = "filename_derived_unverified"
            return fallback
        src[field] = "unavailable"
        return None

    pid = _first(props, "sceneName", "fileID")
    if pid is None and name:
        pid = re.sub(r"\.h5$", "", name, flags=re.I)
    obs["product_id"] = pid
    src["product_id"] = "asf_api" if pid else "unavailable"
    obs["file_name"] = name
    obs["source"] = "ASF Search API (asf_search)"
    obs["url"] = _first(props, "url")
    obs["start_time"] = pick("start_time", ("startTime",), fn.get("start_time"))
    obs["stop_time"] = pick("stop_time", ("stopTime",))
    obs["product_type"] = pick("product_type", ("processingLevel",), fn.get("product_type"))
    obs["platform"] = pick("platform", ("platform",))
    obs["track"] = pick("track", ("pathNumber",), fn.get("track"))
    obs["frame"] = pick("frame", ("frameNumber",), fn.get("frame"))
    obs["direction"] = pick("direction", ("flightDirection",), fn.get("direction"))
    if isinstance(obs["direction"], str):
        obs["direction"] = obs["direction"].upper()
    obs["polarization"] = pick("polarization", ("polarization",))
    obs["cycle"] = fn.get("cycle")
    mat, mat_key = detect_maturity(props)
    obs["maturity"] = mat
    src["maturity"] = f"asf_api:{mat_key}" if mat else "unavailable"
    obs["release"] = pick("release", ("processingRelease", "crid"), fn.get("release"))
    obs["size_bytes"] = _first(props, "bytes")
    obs["md5"] = _first(props, "md5sum")
    bbox = footprint_bbox(record.get("geometry"))
    obs["footprint_bbox"] = list(bbox) if bbox else None
    src["footprint_bbox"] = "asf_api_geometry" if bbox else "unavailable"
    obs["antimeridian_suspect"] = bool(bbox and (bbox[2] - bbox[0]) > 180)
    obs["acquisition_date"] = (parse_time(obs["start_time"]).date().isoformat()
                               if parse_time(obs["start_time"]) else None)
    obs["is_test_fixture"] = str(pid or "").startswith(TEST_FIXTURE_PREFIX)
    obs["field_sources"] = src
    obs["missing_fields"] = [f for f in REQUIRED_FIELDS if obs.get(f) is None]
    obs["raw_properties_json"] = json.dumps(props, default=str, sort_keys=True)
    return obs


def is_qa_or_auxiliary(obs):
    name = (obs.get("file_name") or obs.get("product_id") or "")
    up = name.upper()
    toks = re.split(r"[_.\-]", up)
    if not up.endswith(".H5"):
        return True            # only HDF5 science products belong in the inventory
    return "QA" in toks or "STATS" in toks or "QA_STATS" in up


def dedupe(observations):
    """Deduplicate by actual product_id. Returns (unique, report). Records with no
    product_id are kept out of the inventory and counted."""
    seen, unique, dups, conflicts, no_id = {}, [], 0, [], 0
    for o in observations:
        pid = o.get("product_id")
        if not pid:
            no_id += 1
            continue
        if pid in seen:
            dups += 1
            if seen[pid].get("start_time") != o.get("start_time"):
                conflicts.append(pid)
            continue
        seen[pid] = o
        unique.append(o)
    return unique, {"duplicates_removed": dups, "conflicting_duplicates": conflicts,
                    "records_without_product_id": no_id}
