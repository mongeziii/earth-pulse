"""EARTH PULSE temporal analysis. Pure functions on real extracted arrays.

SIGNATURE RULES (descriptive labels, NOT event classifications, NOT confidence scores).
Let v = mean backscatter (dB) over pixels valid in ALL observations, ordered by time;
steps = diff(v); net = v[-1]-v[0]; thr = minimum meaningful change in dB (user parameter,
default 0.5 dB: an analyst-chosen threshold, not a statistical test); a step is significant
if |step| >= thr; alternations = sign flips between consecutive significant steps.
 1. n < min_obs or common-valid pixels < min_common_valid_pct      -> INSUFFICIENT EVIDENCE
 2. alternations >= 2                                              -> REPEATED VARIATION
 3. range(v) >= thr and |net| < thr (departed and came back)       -> REVERSING PATTERN
 4. |net| < thr and no significant step                            -> NO DETECTABLE CHANGE
 5. |net| >= thr and one step of the same sign carries >= dominance (0.6) of |net|
                                                                   -> SUDDEN CHANGE
      + tag PERSISTENT if >= 2 observations follow and all stay within thr of the post-step level
 6. |net| >= thr otherwise                                         -> GRADUAL CHANGE
 7. anything else                                                  -> NO DETECTABLE CHANGE (noted)
"""
import numpy as np

DEFAULTS = dict(threshold_db=0.5, dominance=0.6, min_obs=3, min_common_valid_pct=10.0)
RULES_TEXT = __doc__


def to_db(a):
    a = np.asarray(a, dtype="float64")
    out = np.full(a.shape, np.nan)
    m = np.isfinite(a) & (a > 0)
    out[m] = 10.0 * np.log10(a[m])
    return out


def series_from_arrays(arrays_linear):
    """arrays_linear: list of 2-D linear-power arrays in time order (same grid).
    Returns per-observation mean/std dB over the COMMON valid mask plus coverage figures."""
    if not arrays_linear:
        raise ValueError("no arrays")
    shp = {np.asarray(a).shape for a in arrays_linear}
    if len(shp) != 1:
        raise ValueError(f"arrays are on different grids {sorted(shp)}; stack is not comparable")
    dbs = [to_db(a) for a in arrays_linear]
    mask = np.all(np.stack([np.isfinite(d) for d in dbs]), axis=0)
    n = mask.size
    out = {"n_obs": len(dbs), "common_valid_pct": round(100.0 * mask.sum() / n, 4),
           "per_obs_valid_pct": [round(100.0 * np.isfinite(d).sum() / n, 4) for d in dbs],
           "mean_db": [], "std_db": [], "common_mask": mask}
    for d in dbs:
        out["mean_db"].append(float(np.mean(d[mask])) if mask.any() else float("nan"))
        out["std_db"].append(float(np.std(d[mask])) if mask.any() else float("nan"))
    return out


def classify_signature(mean_db, common_valid_pct, **overrides):
    p = {**DEFAULTS, **overrides}
    v = np.asarray(mean_db, dtype="float64")
    base = {"label": None, "tags": [], "params": p, "rule": None, "metrics": {}}
    if len(v) < p["min_obs"] or common_valid_pct < p["min_common_valid_pct"] or not np.all(np.isfinite(v)):
        base.update(label="INSUFFICIENT EVIDENCE", rule=(
            f"needs >= {p['min_obs']} observations and >= {p['min_common_valid_pct']}% common valid pixels; "
            f"have {len(v)} observations and {common_valid_pct}%"))
        return base
    thr = p["threshold_db"]
    steps, net = np.diff(v), float(v[-1] - v[0])
    sig = np.abs(steps) >= thr
    signs = np.sign(steps[sig])
    alt = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    rng = float(np.ptp(v))
    k = int(np.argmax(np.abs(steps)))
    dom = float(abs(steps[k]) / abs(net)) if abs(net) > 0 else None
    base["metrics"] = {"net_change_db": net, "steps_db": steps.tolist(), "n_significant_steps": int(sig.sum()),
                       "alternations": alt, "range_db": rng, "largest_step_db": float(steps[k]),
                       "largest_step_index": k, "largest_step_share_of_net": dom}
    if alt >= 2:
        base.update(label="REPEATED VARIATION", rule=f"{alt} direction reversals among significant steps (rule 2)")
    elif rng >= thr and abs(net) < thr:
        base.update(label="REVERSING PATTERN", rule=f"range {rng:.2f} dB >= {thr} but net change {net:.2f} dB < {thr} (rule 3)")
    elif abs(net) < thr and not sig.any():
        base.update(label="NO DETECTABLE CHANGE", rule=f"net change {net:.2f} dB and all steps below {thr} dB (rule 4)")
    elif abs(net) >= thr:
        if sig[k] and dom is not None and dom >= p["dominance"] and np.sign(steps[k]) == np.sign(net):
            base.update(label="SUDDEN CHANGE", rule=f"one step carries {dom:.0%} of the net change (rule 5)")
            post = v[k + 1:]
            if len(post) >= 2 and np.all(np.abs(post - post[0]) < thr):
                base["tags"].append("PERSISTENT")
        else:
            base.update(label="GRADUAL CHANGE", rule="net change exceeds threshold without a dominant single step (rule 6)")
    else:
        base.update(label="NO DETECTABLE CHANGE", rule="consecutive-step variation exceeds threshold but net departure does not (rule 7)")
    return base


def change_story(dates, sig, series, units):
    """Plain-language summary built only from computed values. States what is NOT established."""
    n = len(dates)
    if sig["label"] == "INSUFFICIENT EVIDENCE":
        return (f"The available data are insufficient for a temporal signature: {sig['rule']}. "
                "No change is reported and the data do not establish any cause.")
    m = sig["metrics"]
    tags = f" ({', '.join(sig['tags'])})" if sig["tags"] else ""
    return (f"Between the first ({dates[0]}) and last ({dates[-1]}) available observations, the mean radar "
            f"backscatter over {series['common_valid_pct']:.1f}% of the sampled pixels changed by "
            f"{m['net_change_db']:+.2f} dB (layer units: {units or 'unavailable'}; analysed in dB). "
            f"The pattern was labelled {sig['label']}{tags} using the documented rules "
            f"(threshold {sig['params']['threshold_db']} dB): {sig['rule']}. This is based on {n} valid "
            "observations. The label is descriptive; the data do not establish the underlying cause.")


def evidence_panel(manifest_entries, stack_meta, layer_path, units, stride):
    return {
        "source": "NASA-ISRO NISAR via ASF DAAC (asf_search API); files retrieved with Earthdata auth",
        "product_ids": [e["product_id"] for e in manifest_entries],
        "acquisition_dates": [e["start_time"] for e in manifest_entries],
        "retrieval_status": [e["retrieval_status"] for e in manifest_entries],
        "product_type": stack_meta.get("product_type"), "data_maturity": stack_meta.get("maturity"),
        "processing_release": stack_meta.get("release"), "polarization": stack_meta.get("polarization"),
        "measurement_layer": layer_path, "units": units,
        "processing_method": f"{stack_meta.get('window_description') or f'every {stride}th pixel read from HDF5'}; fill values and non-positive/non-finite "
                             "pixels masked; linear power converted to dB; common valid mask across all dates",
        "calculation_method": "mean dB over common valid pixels per date; steps = consecutive differences; "
                              "labels from documented threshold rules (see ABOUT)",
        "limitations": ["Backscatter change alone cannot identify cause (moisture, vegetation, flooding, "
                        "surface change and processing effects are not separable here).",
                        "Decimated sampling; thresholds are analyst parameters, not statistical confidence.",
                        "Only compatible observations (same track, frame, direction, polarization, maturity, "
                        "release) are combined."]}
