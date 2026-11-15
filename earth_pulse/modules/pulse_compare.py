"""Compare Two Earths: gate comparisons between two observation stacks."""
from .observation_ranking import bbox_overlap
from .product_metadata import parse_time

BLOCK_FIELDS = ("product_type", "polarization", "maturity", "release", "layer_path_leaf", "units")
WARN_FIELDS = ("direction", "track")


def compare_stacks(a, b):
    """a, b: dicts with stack metadata (product_type, polarization, maturity, release, direction,
    track, layer_path_leaf, units, bbox, first_acquisition, last_acquisition, n_observations).
    Returns {allowed, blocking, warnings, mode, spatial_overlap, temporal_overlap_days}."""
    blocking, warnings = [], []
    for f in BLOCK_FIELDS:
        va, vb = a.get(f), b.get(f)
        if f in ("maturity", "release", "polarization") and str(va).startswith("UNKNOWN") and va == vb:
            warnings.append(f"{f} is not provided by the API for either stack: compatibility on {f} is UNVERIFIED")
            continue
        if va is None or vb is None:
            blocking.append(f"{f} unknown for {'both stacks' if va is None and vb is None else 'one stack'}")
        elif va != vb:
            blocking.append(f"{f} differs: {va} vs {vb}")
    mode = "absolute levels and trajectories"
    for f in WARN_FIELDS:
        if a.get(f) != b.get(f):
            warnings.append(f"{f} differs ({a.get(f)} vs {b.get(f)}): backscatter levels depend on acquisition "
                            "geometry, so only change relative to each stack's first observation is comparable")
            mode = "relative change only (dB relative to first observation)"
    ov = None
    if a.get("bbox") and b.get("bbox"):
        ov = bbox_overlap(a["bbox"], b["bbox"])
    ta0, ta1 = parse_time(a.get("first_acquisition")), parse_time(a.get("last_acquisition"))
    tb0, tb1 = parse_time(b.get("first_acquisition")), parse_time(b.get("last_acquisition"))
    t_ov = None
    if all((ta0, ta1, tb0, tb1)):
        t_ov = max(0.0, (min(ta1, tb1) - max(ta0, tb0)).total_seconds() / 86400)
        if t_ov == 0:
            warnings.append("the stacks do not overlap in time; differences may reflect different periods")
    if ov is not None and ov > 0.1:
        warnings.append("the stacks spatially overlap; they are not independent places")
    return {"allowed": not blocking, "blocking": blocking, "warnings": warnings,
            "mode": mode if not blocking else None, "spatial_overlap": ov, "temporal_overlap_days": t_ov}
