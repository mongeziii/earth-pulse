"""Thin wrapper over the official ASF Search API (asf_search) and NASA CMR.
Raises ArchiveAccessError with the exact failure so callers never fake success."""
import json
from datetime import datetime

from .config import NISAR_DATASET, PRODUCT_TYPE
from .product_metadata import footprint_bbox

MAX_VALUE_CHARS = 300   # drop long values (nested dicts, big lists) to keep memory small


class ArchiveAccessError(Exception):
    pass


def _asf():
    try:
        import asf_search as asf
        return asf
    except ImportError as e:
        raise ArchiveAccessError("asf_search is not installed (pip install -r requirements.txt).") from e


def slim_record(properties, geometry):
    """Keep scalar/short properties verbatim; store the footprint as its bounding box polygon.
    Large nested metadata is dropped (memory); nothing is altered or invented."""
    props = {k: v for k, v in dict(properties).items()
             if v is None or isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= MAX_VALUE_CHARS)}
    bb = footprint_bbox(geometry) if geometry else None
    geom = None
    if bb:
        x0, y0, x1, y1 = bb
        geom = {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
                "note": "bounding box of the original footprint"}
    return {"properties": props, "geometry": geom}


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def raw_search(start: datetime, end: datetime, cap: int, wkt=None, extra=None, session=None):
    """One ASF query. Returns (records, search_complete).
    records: list of {"properties": dict, "geometry": dict|None}  (unmodified API content)
    search_complete: bool|None  (None = the library did not say)"""
    asf = _asf()
    kwargs = dict(dataset=NISAR_DATASET, processingLevel=PRODUCT_TYPE,
                  start=_iso(start), end=_iso(end), maxResults=cap)
    if wkt:
        kwargs["intersectsWith"] = wkt
    kwargs.update(extra or {})
    if session is not None:
        kwargs["opts"] = asf.ASFSearchOptions(session=session)
    try:
        results = asf.search(**kwargs)
    except Exception as e:
        raise ArchiveAccessError(f"ASF search failed: {type(e).__name__}: {e}") from e
    complete = getattr(results, "searchComplete", None)
    records = [slim_record(p.properties, getattr(p, "geometry", None)) for p in results]
    del results
    return records, complete


def cmr_nisar_collections(timeout=30):
    """List NISAR collections known to NASA CMR (to investigate Harmony/service support).
    Reports only fields CMR actually returns; does not assert Harmony support."""
    import requests
    try:
        r = requests.get("https://cmr.earthdata.nasa.gov/search/collections.json",
                         params={"keyword": "NISAR", "page_size": 200}, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise ArchiveAccessError(f"CMR query failed: {type(e).__name__}: {e}") from e
    return [{"concept_id": c.get("id"), "short_name": c.get("short_name"),
             "version_id": c.get("version_id"), "title": c.get("title"),
             "has_granules": c.get("has_granules"), "associations": c.get("associations")}
            for c in r.json().get("feed", {}).get("entry", [])]
