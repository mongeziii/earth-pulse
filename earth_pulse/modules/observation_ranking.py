"""Group observations into compatible stacks and rank them with a transparent rule.

Stack key = (product_type, track, direction, frame, polarization, maturity, release).
Observations with any unknown key field cannot be placed in a stack (never assumed equal).

Ranking (lexicographic, no hidden weights):
 1. eligible stacks first (>= MIN_STACK_SIZE distinct acquisitions, overlap >= MIN_BBOX_OVERLAP)
 2. more distinct acquisition dates
 3. smaller maximum temporal gap
 4. higher minimum footprint overlap
 5. higher measured valid-pixel % (only once measured; unmeasured does not rank)
"""
from itertools import combinations
from statistics import median

from .config import MIN_STACK_SIZE, MIN_BBOX_OVERLAP
from .data_validation import check_date_order
from .product_metadata import parse_time

KEY_FIELDS = ("product_type", "track", "direction", "frame", "polarization", "maturity", "release")


def bbox_overlap(a, b):
    """Intersection area / smaller bbox area (planar lon/lat; adequate for same-frame footprints)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return 0.0 if small <= 0 else (ix * iy) / small


UNKNOWN_TEXT = "UNKNOWN (not provided by API)"


def stack_key(o, fields=KEY_FIELDS):
    vals = tuple(o.get(f) for f in fields)
    return vals if all(v is not None for v in vals) else None


def build_stacks(observations, valid_pixel_pct=None, ignore_fields=()):
    """valid_pixel_pct: optional {product_id: pct} from real extraction.
    ignore_fields: key fields the API does not supply; grouping then does NOT use them and the stack
    reports them as UNKNOWN. This is an explicit, labelled loosening - never the default."""
    valid_pixel_pct = valid_pixel_pct or {}
    used = tuple(f for f in KEY_FIELDS if f not in ignore_fields)
    groups, unstackable = {}, []
    for o in observations:
        k = stack_key(o, used)
        (groups.setdefault(k, []) if k else unstackable).append(o)
    rows = []
    for k, members in groups.items():
        ordered, _ = check_date_order(members)
        by_date = {}
        for o in ordered:                       # one observation per acquisition timestamp
            by_date.setdefault(o["start_time"], o)
        ordered = list(by_date.values())
        times = [parse_time(o["start_time"]) for o in ordered]
        gaps = [(b - a).total_seconds() / 86400 for a, b in zip(times, times[1:])]
        boxes = [o["footprint_bbox"] for o in ordered if o.get("footprint_bbox")]
        ov = min((bbox_overlap(a, b) for a, b in combinations(boxes, 2)), default=None) \
            if len(boxes) == len(ordered) and len(boxes) > 1 else None
        vp = [valid_pixel_pct[o["product_id"]] for o in ordered if o["product_id"] in valid_pixel_pct]
        rows.append({
            "stack_id": "|".join(map(str, k)),
            **{f: (k[used.index(f)] if f in used else UNKNOWN_TEXT) for f in KEY_FIELDS},
            "unverified_fields": ";".join(ignore_fields),
            "n_observations": len(ordered),
            "first_acquisition": ordered[0]["start_time"], "last_acquisition": ordered[-1]["start_time"],
            "span_days": round((times[-1] - times[0]).total_seconds() / 86400, 3) if len(times) > 1 else 0.0,
            "median_gap_days": round(median(gaps), 3) if gaps else None,
            "max_gap_days": round(max(gaps), 3) if gaps else None,
            "min_footprint_overlap": round(ov, 4) if ov is not None else None,
            "mean_valid_pixel_pct": round(sum(vp) / len(vp), 3) if vp else None,
            "product_ids": ";".join(o["product_id"] for o in ordered),
            "eligible": len(ordered) >= MIN_STACK_SIZE and ov is not None and ov >= MIN_BBOX_OVERLAP,
        })
    rows.sort(key=lambda r: (not r["eligible"], -r["n_observations"],
                             r["max_gap_days"] if r["max_gap_days"] is not None else 1e9,
                             -(r["min_footprint_overlap"] or 0), -(r["mean_valid_pixel_pct"] or 0)))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows, {"unstackable_count": len(unstackable),
                  "unstackable_product_ids": [o.get("product_id") for o in unstackable]}


def explain_selection(rows):
    """Plain-language reason built only from the computed metrics."""
    if not rows:
        return "No stacks could be built: no observations had complete stack-key metadata."
    best = rows[0]
    base = (f"Stack rank 1: track {best['track']}, frame {best['frame']}, {best['direction']}, "
            f"polarization {best['polarization']}, maturity {best['maturity']}, release {best['release']}; "
            f"{best['n_observations']} distinct acquisitions from {best['first_acquisition']} to "
            f"{best['last_acquisition']}, maximum gap {best['max_gap_days']} days, "
            f"minimum footprint overlap {best['min_footprint_overlap']}.")
    if best["eligible"]:
        return base + f" It meets the minimum of {MIN_STACK_SIZE} compatible observations. " \
                      "Valid-pixel coverage is not yet measured unless extraction has been run."
    return ("NO stack meets the minimum criteria (>= %d compatible observations with footprint "
            "overlap >= %.2f). Strongest available alternative: " % (MIN_STACK_SIZE, MIN_BBOX_OVERLAP)) + base
