"""Build and load locally cached, verified observation stacks (data/processed/<stack>/).
Origin of every stored array is a downloaded NASA product file; the loader refuses anything
else (missing provenance, failed retrieval, test fixtures)."""
import json
import re
from pathlib import Path

import numpy as np

from .config import PROCESSED_DIR
from .data_extraction import download_product, find_gcov_layers, read_layer
from .data_validation import check_date_order

OK_STATUSES = ("DOWNLOADED_VERIFIED", "CACHED_VERIFIED",
               "DOWNLOADED_UNVERIFIED_NO_CHECKSUM", "CACHED_UNVERIFIED_NO_CHECKSUM")


def slug(stack_id):
    return re.sub(r"[^A-Za-z0-9]+", "_", stack_id).strip("_")[:120]


def build_stack(stack_row, observations, session, layer_term=None, stride=8, max_products=12, root=PROCESSED_DIR):
    """Download + read every product of a ranked stack. Returns the manifest dict (also saved)."""
    ids = stack_row["product_ids"].split(";")[:max_products]
    obs, _ = check_date_order([o for o in observations if o["product_id"] in ids])
    d = Path(root) / slug(stack_row["stack_id"]); d.mkdir(parents=True, exist_ok=True)
    entries, term = [], layer_term
    for o in obs:
        path, status = download_product(o, session)
        e = {"product_id": o["product_id"], "start_time": o["start_time"], "url": o["url"],
             "retrieval_status": status, "origin": "NASA_PRODUCT_FILE", "is_test_fixture": False}
        if path:
            layers = [l for l in find_gcov_layers(path) if l["diagonal"]]
            term = term or (layers[0]["polarization_term"] if layers else None)
            match = [l for l in layers if l["polarization_term"] == term]
            if not match:
                e["retrieval_status"] = f"FAILED: layer {term} not in product"
            else:
                r = read_layer(path, match[0]["path"], stride)
                np.save(d / f"{o['product_id']}.npy", r["array"].astype("float32"))
                e.update(layer_path=r["layer_path"], units=r["units"], stride=stride, shape=list(r["array"].shape),
                         quality=r["quality"], array_file=f"{o['product_id']}.npy", layer_term=term)
        entries.append(e)
    manifest = {"stack_id": stack_row["stack_id"], "stack": {k: stack_row.get(k) for k in
                ("product_type", "track", "direction", "frame", "polarization", "maturity", "release",
                 "first_acquisition", "last_acquisition", "n_observations")},
                "bbox": _union_bbox(obs), "entries": entries}
    (d / "stack_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _union_bbox(obs):
    bs = [o["footprint_bbox"] for o in obs if o.get("footprint_bbox")]
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)] if bs else None


def list_stacks(root=PROCESSED_DIR):
    return sorted(p.parent.name for p in Path(root).glob("*/stack_manifest.json"))


def load_stack(name, root=PROCESSED_DIR):
    """Returns (manifest, entries_ok, arrays, skipped[{product_id, reason}])."""
    d = Path(root) / name
    m = json.loads((d / "stack_manifest.json").read_text())
    ok, arrays, skipped = [], [], []
    for e in m["entries"]:
        if e.get("is_test_fixture") or e.get("origin") != "NASA_PRODUCT_FILE":
            reason = "not an NASA product file (refused)"
        elif e.get("retrieval_status") not in OK_STATUSES:
            reason = f"retrieval status {e.get('retrieval_status')}"
        elif not (d / e.get("array_file", "")).is_file():
            reason = "array file missing"
        else:
            ok.append(e); arrays.append(np.load(d / e["array_file"]).astype("float64")); continue
        skipped.append({"product_id": e.get("product_id"), "reason": reason})
    return m, ok, arrays, skipped
