"""Central configuration. Values here are search/analysis parameters, never data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "metadata"

NISAR_DATASET = "NISAR"
PRODUCT_TYPE = "GCOV"
SEARCH_CAP = 1000            # a response this large is treated as possibly truncated
MIN_SPLIT_SPAN_HOURS = 6     # smallest time window we will bisect to
MIN_STACK_SIZE = 3           # minimum compatible observations for a temporal demo
MIN_BBOX_OVERLAP = 0.90      # intersection / smaller bbox area; tunable, documented
# Tokens searched for (verbatim) in API metadata to detect data maturity.
MATURITY_TOKENS = ("VALIDATED", "PROVISIONAL", "BETA")
DEFAULT_SEARCH_START = "2025-07-01"   # conservative lower bound for NISAR archive search


def ensure_dirs():
    for d in (RAW_DIR, PROCESSED_DIR, META_DIR):
        d.mkdir(parents=True, exist_ok=True)
