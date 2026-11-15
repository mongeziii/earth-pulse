"""Evidence-grounded extensions. No generated satellite measurements or external AI claims."""
from datetime import datetime, timezone
import numpy as np


def center_from_bbox(bbox):
    """Return (lat, lon) only for verifiable numeric geographic bounding boxes."""
    if isinstance(bbox, dict):
        if all(k in bbox for k in ('minlon','minlat','maxlon','maxlat')):
            v = [bbox[k] for k in ('minlon','minlat','maxlon','maxlat')]
        else:
            return None
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        v = bbox
    else:
        return None
    try:
        a,b,c,d = map(float,v)
        if not (-180 <= a <= c <= 180 and -90 <= b <= d <= 90): return None
        return ((b+d)/2,(a+c)/2)
    except (TypeError,ValueError):
        return None


def observation_report(ok, arrays, threshold_db=1.0):
    """Produce traceable measured values and bounded interpretation."""
    from modules.pulse_analysis import series_from_arrays, classify_signature, change_story
    if not ok or len(ok) != len(arrays):
        raise ValueError('Verified observations and arrays must match')
    series = series_from_arrays(arrays)
    dates = [e['start_time'] for e in ok]
    sig = classify_signature(series['mean_db'], series['common_valid_pct'], threshold_db=threshold_db)
    return {
        'source': 'NASA-ISRO NISAR products retrieved through ASF DAAC',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'measurement': 'mean radar backscatter (dB) across common valid sampled pixels',
        'observations': [{'acquired_utc': e['start_time'], 'product_id': e['product_id'],
                          'mean_db': float(y)} for e,y in zip(ok, series['mean_db'])],
        'signature': sig['label'], 'signature_rule': sig['rule'],
        'threshold_db': threshold_db,
        'story': change_story(dates, sig, series, ok[0].get('units')),
        'limitations': ['Pattern classification is not an event or cause identification.',
                        'No values are interpolated between observed dates.',
                        'Backscatter may respond to geometry, moisture, vegetation and acquisition conditions.',
                        'The chosen threshold is an analyst parameter, not a calibrated confidence probability.']
    }


def new_products(previous_ids, current_ids):
    return sorted(set(current_ids) - set(previous_ids))
