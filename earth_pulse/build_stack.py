"""Download + extract the best (or chosen) ranked stack for the Phase 2 app.
  python build_stack.py --rank 1 --max-products 12 [--layer HHHH] [--stride 8]
Needs a prior `python run_phase1.py --refresh` and Earthdata credentials. GCOV files are large."""
import argparse
import csv
import json
import sys

from modules import config as C
from modules.earthdata_auth import build_asf_session
from modules.stack_store import build_stack

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=1)
ap.add_argument("--max-products", type=int, default=12)
ap.add_argument("--layer", default=None)
ap.add_argument("--stride", type=int, default=8)
a = ap.parse_args()
try:
    stacks = list(csv.DictReader(open(C.META_DIR / "candidate_stacks.csv")))
    inv = list(csv.DictReader(open(C.META_DIR / "observation_inventory.csv")))
except FileNotFoundError:
    sys.exit("Run `python run_phase1.py --refresh` first (no stack inventory found).")
row = next((s for s in stacks if int(s["rank"]) == a.rank), None)
if not row:
    sys.exit(f"No stack with rank {a.rank}.")
for o in inv:  # CSV -> types the builder needs
    o["footprint_bbox"] = json.loads(o["footprint_bbox"]) if o.get("footprint_bbox") else None
    o["size_bytes"] = o.get("size_bytes") or None
    o["md5"] = o.get("md5") or None
m = build_stack(row, inv, build_asf_session(), a.layer, a.stride, a.max_products)
for e in m["entries"]:
    print(e["product_id"], e["retrieval_status"])
