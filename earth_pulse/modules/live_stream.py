"""LIVE, IN-MEMORY NASA data access. Nothing in this module writes data to disk.

discover()      ASF Search API -> observations + ranked stacks, held in memory only.
stream_stack()  opens each NISAR product remotely over HTTPS (range reads, no file saved) and reads a
                small window around the area of interest straight into numpy arrays.

Everything that touches NASA services is UNVERIFIED until run with real credentials: the HDF5 layout
assumptions (coordinates, projection EPSG) are checked at run time and fail with an explicit message."""
from datetime import datetime, timezone

import numpy as np

from . import nasa_api
from .aoi import parse_aoi
from .config import SEARCH_CAP
from .data_validation import validate_array
from .nisar_search import search_archive
from .observation_ranking import build_stacks, explain_selection
from .product_metadata import dedupe, is_qa_or_auxiliary, normalize_record


FILE_VERIFIABLE_FIELDS = ("polarization",)


class StreamError(Exception):
    pass


# ------------------------------------------------------------------ discovery (in memory)
def discover(aoi_text, start, end, ignore_fields=(), max_calls=60, on_window=None,
             search_fn=None, _allow_test_fixtures=False):
    """aoi_text: 'lon,lat' or 'minlon,minlat,maxlon,maxlat'. start/end: aware datetimes."""
    label, wkt = parse_aoi(aoi_text)
    fn = search_fn or (lambda s, e, cap: nasa_api.raw_search(s, e, cap, wkt=wkt))
    span = max(1, (end - start).days + 1)
    records, report = search_archive(fn, start, end, span, SEARCH_CAP, max_calls, on_window)
    obs = [normalize_record(r) for r in records]
    fx = [o["product_id"] for o in obs if o["is_test_fixture"]]
    if fx and not _allow_test_fixtures:
        raise RuntimeError(f"Refusing test-fixture records in the live pipeline: {fx[:3]}")
    for o in obs:
        o["qa_or_auxiliary"] = is_qa_or_auxiliary(o)
    science = [o for o in obs if not o["qa_or_auxiliary"]]
    unique, dd = dedupe(science)
    # A field that is empty for EVERY product and can be verified from the file contents at streaming time
    # (the layer actually read, e.g. HHHH) is left out of the grouping key. Labelled; never silent.
    auto = [f for f in FILE_VERIFIABLE_FIELDS if unique and all(f in o["missing_fields"] for o in unique)
            and f not in ignore_fields]
    ignore_fields = tuple(ignore_fields) + tuple(auto)
    stacks, info = build_stacks(unique, ignore_fields=ignore_fields)
    missing = {}
    for o in unique:
        for f in o["missing_fields"]:
            missing[f] = missing.get(f, 0) + 1
    return {"aoi_text": aoi_text.strip(), "aoi_label": label, "wkt": wkt, "observations": unique, "stacks": stacks, "report": report,
            "dedupe": dd, "excluded_qa_or_auxiliary": len(obs) - len(science),
            "missing_field_counts": missing, "unstackable": info["unstackable_count"],
            "selection_reason": explain_selection(stacks),
            "sample_property_keys": sorted(records[0]["properties"]) if records else [],
            "sample_properties": dict(sorted(records[0]["properties"].items())) if records else {},
            "sample_normalized": ({k: v for k, v in normalize_record(records[0]).items() if k != "raw_properties_json"}
                                  if records else {}),
            "ignored_fields": list(ignore_fields), "auto_ignored_fields": auto,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "origin": "LIVE_API (memory only, nothing saved)"}


# ------------------------------------------------------------------ window geometry (pure)
def window_indices(x, y, X, Y, half_px):
    """Pixel window (y0, y1, x0, x1) around projected point (X, Y) on 1-D coordinate vectors."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        raise StreamError("coordinate vectors too short")
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    if not (x.min() - dx <= X <= x.max() + dx and y.min() - dy <= Y <= y.max() + dy):
        raise StreamError("area of interest lies outside this product's footprint")
    ix, iy = int(np.argmin(np.abs(x - X))), int(np.argmin(np.abs(y - Y)))
    x0, x1 = max(0, ix - half_px), min(len(x), ix + half_px + 1)
    y0, y1 = max(0, iy - half_px), min(len(y), iy + half_px + 1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise StreamError("window at the product edge is too small (<16 px); choose an area further inside")
    return y0, y1, x0, x1


def lonlat_to_xy(epsg, lon, lat):
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise StreamError("pyproj is not installed (pip install -r requirements.txt)") from e
    return Transformer.from_crs(4326, int(epsg), always_xy=True).transform(lon, lat)


# ------------------------------------------------------------------ remote reading
def _scalar(v):
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    a = np.asarray(v).ravel()
    return a[0].item() if a.size else None


def read_window(fobj, lon, lat, half_px=300, layer_term=None, stride=1):
    """Read a window of one real, diagonal covariance layer from an open (file-like) NISAR HDF5."""
    import h5py
    with h5py.File(fobj, "r") as f:
        layers = []
        def visit(name, obj):
            leaf = name.rsplit("/", 1)[-1]
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and len(leaf) == 4 and set(leaf) <= set("HVLR") \
                    and leaf[:2] == leaf[2:] and not np.issubdtype(obj.dtype, np.complexfloating):
                layers.append((leaf, name))
        f.visititems(visit)
        if not layers:
            raise StreamError("no real-valued diagonal covariance layers (e.g. HHHH) found in this product")
        pick = [l for l in layers if l[0] == layer_term] if layer_term else layers[:1]
        if not pick:
            raise StreamError(f"layer {layer_term} not in product (found {[l[0] for l in layers]})")
        term, path = pick[0]
        ds = f[path]
        grp = f[path.rsplit("/", 1)[0]]
        names = sorted(grp.keys())
        try:
            x, y = grp["xCoordinates"][:], grp["yCoordinates"][:]
        except KeyError:
            raise StreamError(f"xCoordinates/yCoordinates not found next to the layer; group has {names}")
        proj = grp.get("projection")
        epsg = None
        if proj is not None:
            epsg = _scalar(proj.attrs.get("epsg"))
            if epsg is None:
                try:
                    v = _scalar(proj[()])
                    epsg = v if isinstance(v, (int, float)) and v > 1000 else None
                except Exception:
                    epsg = None
        if epsg is None:
            raise StreamError(f"projection EPSG not found; group has {names}")
        X, Y = lonlat_to_xy(int(epsg), lon, lat)
        y0, y1, x0, x1 = window_indices(x, y, X, Y, half_px)
        arr = ds[y0:y1:stride, x0:x1:stride].astype("float64")
        fill, units = _scalar(ds.attrs.get("_FillValue")), _scalar(ds.attrs.get("units"))
        q = validate_array(arr, fill_value=fill)
        if fill is not None and not np.isnan(fill):
            arr = np.where(arr == fill, np.nan, arr)
        sig = {"epsg": int(epsg), "x_first": float(x[x0]), "y_first": float(y[y0]),
               "dx": float(x[1] - x[0]), "dy": float(y[1] - y[0]), "shape": list(arr.shape), "stride": stride}
    return {"array": arr, "units": units, "layer_path": path, "layer_term": term, "quality": q, "grid": sig}


def earthaccess_open(url):
    """Open a protected NASA https URL as a file-like object (range reads; nothing saved to disk)."""
    from .earthdata_auth import earthaccess_login
    earthaccess_login()
    import earthaccess
    files = earthaccess.open([url])
    if not files:
        raise StreamError("earthaccess returned no file handle for this URL")
    return files[0]


def _same_grid(a, b):
    return (a["epsg"] == b["epsg"] and a["shape"] == b["shape"] and a["stride"] == b["stride"]
            and all(np.isclose(a[k], b[k], rtol=0, atol=1e-6) for k in ("x_first", "y_first", "dx", "dy")))


def stream_stack(stack_row, observations, lon, lat, half_px=300, stride=1, max_products=16, layer_term=None,
                 opener=earthaccess_open, reader=read_window, progress=None):
    """Stream the most recent `max_products` observations of a ranked stack. Returns
    (manifest, ok_entries, arrays, skipped). No file is written. opener/reader are injectable for tests."""
    ids = set(stack_row["product_ids"].split(";"))
    obs = sorted([o for o in observations if o["product_id"] in ids], key=lambda o: o["start_time"])[-max_products:]
    entries, arrays, skipped, ref, term = [], [], [], None, layer_term
    for i, o in enumerate(obs):
        if progress:
            progress(i, len(obs), o)
        f = None
        try:
            if not o.get("url"):
                raise StreamError("record has no URL")
            f = opener(o["url"])
            r = reader(f, lon, lat, half_px, term, stride)
            term = term or r["layer_term"]
            if ref is None:
                ref = r["grid"]
            elif not _same_grid(ref, r["grid"]):
                raise StreamError("window grid differs from the first observation; excluded to avoid mixing grids")
            if not r["quality"]["ok"]:
                raise StreamError("window failed data-quality checks: " + "; ".join(r["quality"]["problems"] or ["no usable pixels"]))
        except Exception as ex:
            skipped.append({"product_id": o["product_id"], "reason": f"{type(ex).__name__}: {ex}"})
            continue
        finally:
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        entries.append({"product_id": o["product_id"], "start_time": o["start_time"], "url": o["url"],
                        "retrieval_status": "STREAMED_NOT_STORED", "origin": "NASA_PRODUCT_STREAM",
                        "is_test_fixture": False, "layer_term": r["layer_term"], "layer_path": r["layer_path"],
                        "units": r["units"], "stride": stride, "shape": r["grid"]["shape"], "quality": r["quality"]})
        arrays.append(r["array"])
    boxes = [o["footprint_bbox"] for o in obs if o.get("footprint_bbox")]
    desc = (f"window of +/-{half_px} pixels (stride {stride}) around the area of interest, streamed over HTTPS "
            "from the NASA archive (no file saved)")
    manifest = {"stack_id": stack_row["stack_id"],
                "stack": {**{k: stack_row.get(k) for k in ("product_type", "track", "direction", "frame", "polarization",
                                                              "maturity", "release", "first_acquisition",
                                                              "last_acquisition", "n_observations", "unverified_fields")},
                          "window_description": desc},
                "polarization_from_file": term,
                "bbox": [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                         max(b[3] for b in boxes)] if boxes else None,
                "aoi": {"lon": lon, "lat": lat}, "grid": ref,
                "streamed_at_utc": datetime.now(timezone.utc).isoformat()}
    if term and str(manifest["stack"].get("polarization", "")).startswith("UNKNOWN") and entries:
        manifest["stack"]["polarization"] = f"{term} (read from files)"
        manifest["stack"]["unverified_fields"] = ";".join(
            f for f in (manifest["stack"].get("unverified_fields") or "").split(";") if f and f != "polarization")
    return manifest, entries, arrays, skipped
