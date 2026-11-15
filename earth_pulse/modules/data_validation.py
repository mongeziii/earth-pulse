"""Metadata and array validation, date ordering and compatibility checks."""
import numpy as np

from .product_metadata import parse_time

COMPAT_FIELDS = ("product_type", "polarization", "track", "frame", "direction", "maturity", "release")


def validate_metadata(obs):
    return {"product_id": obs.get("product_id"), "missing_fields": list(obs.get("missing_fields", [])),
            "complete": not obs.get("missing_fields"), "is_test_fixture": bool(obs.get("is_test_fixture")),
            "antimeridian_suspect": bool(obs.get("antimeridian_suspect"))}


def validate_array(arr, fill_value=None):
    """Quality statistics for a 2-D GCOV power array (linear units, must be >= 0 and finite)."""
    a = np.asarray(arr, dtype="float64")
    total = int(a.size)
    if total == 0:
        return {"total_pixels": 0, "valid_pixel_pct": 0.0, "missing_pct": 100.0, "ok": False,
                "problems": ["empty array"]}
    if fill_value is not None and not np.isnan(fill_value):
        a = np.where(a == fill_value, np.nan, a)
    finite = np.isfinite(a)
    n_finite = int(finite.sum())
    neg = int((a[finite] < 0).sum())
    zero = int((a[finite] == 0).sum())
    usable = n_finite - neg - zero          # finite and strictly positive (log-able)
    problems = []
    if n_finite == 0: problems.append("no finite values")
    if neg: problems.append(f"{neg} negative values (invalid for power)")
    if n_finite and np.nanstd(a[finite]) == 0: problems.append("constant array")
    return {"total_pixels": total, "finite_pixels": n_finite, "zero_pixels": zero,
            "negative_pixels": neg, "usable_pixels": usable,
            "valid_pixel_pct": round(100.0 * usable / total, 4),
            "missing_pct": round(100.0 * (total - n_finite) / total, 4),
            "ok": usable > 0 and neg == 0 and "constant array" not in problems,
            "problems": problems}


def check_date_order(observations):
    """Sort by acquisition time. Returns (sorted, report). Undated records are excluded."""
    dated = [(parse_time(o.get("start_time")), o) for o in observations]
    undated = [o.get("product_id") for t, o in dated if t is None]
    ok = sorted([(t, o) for t, o in dated if t is not None], key=lambda x: x[0])
    ts = [t for t, _ in ok]
    return [o for _, o in ok], {"undated_product_ids": undated,
                                "duplicate_timestamps": len(ts) - len(set(ts)),
                                "n_ordered": len(ts)}


def compatibility(a, b):
    """(compatible, reasons). Unknown values count as incompatible: we never assume a match."""
    reasons = []
    for f in COMPAT_FIELDS:
        va, vb = a.get(f), b.get(f)
        if va is None or vb is None:
            reasons.append(f"{f}: unknown for {'both' if va is None and vb is None else 'one'} product(s)")
        elif va != vb:
            reasons.append(f"{f}: {va} != {vb}")
    return (not reasons), reasons


def build_quality_report(observations, dedupe_report, search_report, array_checks=None):
    n = len(observations)
    return {"n_observations": n,
            "n_complete_metadata": sum(1 for o in observations if not o.get("missing_fields")),
            "missing_field_counts": {f: sum(1 for o in observations if f in o.get("missing_fields", []))
                                     for f in sorted({f for o in observations for f in o.get("missing_fields", [])})},
            "dedupe": dedupe_report,
            "search_complete": search_report.get("complete"),
            "unresolved_truncated_windows": search_report.get("unresolved_truncated_windows", []),
            "antimeridian_suspect_count": sum(1 for o in observations if o.get("antimeridian_suspect")),
            "array_checks": array_checks or {}}
