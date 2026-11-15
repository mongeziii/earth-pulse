"""Download/verify NISAR products and read measurement layers from HDF5.
h5py is imported lazily. Layer discovery walks the real file; it does not assume a layout.
(Expected GCOV layout per NISAR docs: science/LSAR/GCOV/grids/frequencyA/<HHHH|HVHV|...>;
this must be CONFIRMED with `inspect_h5` on the first real file.)"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import RAW_DIR, META_DIR
from .data_validation import validate_array

LOG_PATH = META_DIR / "retrieval_log.json"


def _h5py():
    try:
        import h5py
        return h5py
    except ImportError as e:
        raise RuntimeError("h5py is not installed (pip install -r requirements.txt).") from e


def _log(entry):
    log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    log.append({**entry, "time_utc": datetime.now(timezone.utc).isoformat()})
    LOG_PATH.write_text(json.dumps(log, indent=2))


def _md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download_product(obs, session, dest_dir=RAW_DIR):
    """Download one product (GCOV files can be GB-sized: only call for selected stacks).
    Verifies size and md5 when the archive supplies them. Returns (path|None, status)."""
    import requests  # noqa: F401  (ASFSession is a requests.Session)
    url, name = obs.get("url"), obs.get("file_name")
    if not url or not name:
        return None, "FAILED: record has no url/file name"
    dest = Path(dest_dir) / name
    checked = bool(obs.get("size_bytes") or obs.get("md5"))
    tag = "VERIFIED" if checked else "UNVERIFIED_NO_CHECKSUM"   # never claim a check that did not happen
    def verified(p):
        if obs.get("size_bytes") and p.stat().st_size != int(obs["size_bytes"]):
            return False
        return not obs.get("md5") or _md5(p) == obs["md5"]
    if dest.exists() and verified(dest):
        _log({"product_id": obs["product_id"], "status": f"CACHED_{tag}", "url": url})
        return dest, f"CACHED_{tag}"
    part = dest.with_suffix(".part")
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        part.replace(dest)
    except Exception as e:
        _log({"product_id": obs["product_id"], "status": f"FAILED: {type(e).__name__}", "url": url})
        return None, f"FAILED: {type(e).__name__}: {e}"
    if not verified(dest):
        dest.unlink(missing_ok=True)
        _log({"product_id": obs["product_id"], "status": "FAILED: size/md5 mismatch", "url": url})
        return None, "FAILED: size/md5 mismatch"
    _log({"product_id": obs["product_id"], "status": f"DOWNLOADED_{tag}", "url": url})
    return dest, f"DOWNLOADED_{tag}"


def inspect_h5(path, max_items=400):
    """Walk an HDF5 file; return groups/datasets with shape, dtype and key attributes."""
    h5py = _h5py()
    items = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if len(items) < max_items:
                items.append({"path": name, "kind": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                              "shape": list(obj.shape) if isinstance(obj, h5py.Dataset) else None,
                              "dtype": str(obj.dtype) if isinstance(obj, h5py.Dataset) else None,
                              "units": _attr(obj, "units")})
        f.visititems(visit)
    return items


def _attr(obj, key):
    v = obj.attrs.get(key)
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else None
    return v.item() if hasattr(v, "item") else v


def find_gcov_layers(path):
    """Discover real 2-D covariance layers (names of 4 letters from H/V/L/R) in the file."""
    h5py = _h5py()
    layers = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            leaf = name.rsplit("/", 1)[-1]
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and re.fullmatch(r"[HVLR]{4}", leaf):
                layers.append({"path": name, "polarization_term": leaf, "shape": list(obj.shape),
                               "dtype": str(obj.dtype), "units": _attr(obj, "units"),
                               "diagonal": leaf[:2] == leaf[2:]})
        f.visititems(visit)
    return layers


def read_layer(path, layer_path, stride=8):
    """Read a decimated (every `stride`-th pixel) window of one layer. Returns dict with the
    array (NaN = missing), quality stats and provenance. Complex off-diagonal layers rejected."""
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        ds = f[layer_path]
        if np.issubdtype(ds.dtype, np.complexfloating):
            raise ValueError("Complex-valued layer: not a real power measurement; not read.")
        arr = ds[::stride, ::stride].astype("float64")
        fill = _attr(ds, "_FillValue")
        units = _attr(ds, "units")
    stats = validate_array(arr, fill_value=fill)
    if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
        arr = np.where(arr == fill, np.nan, arr)
    return {"array": arr, "units": units, "fill_value": fill, "stride": stride,
            "layer_path": layer_path, "quality": stats, "source_file": str(path)}
