"""Harmony investigation (Phase 1.6). Reports facts only; does NOT assume NISAR subsetting works."""
from .nasa_api import cmr_nisar_collections


def harmony_status():
    try:
        import harmony  # harmony-py
        installed = getattr(harmony, "__version__", "installed")
    except ImportError:
        installed = None
    return {"harmony_py_installed": installed,
            "nisar_harmony_support": "UNVERIFIED",
            "how_to_verify": "Run `python run_phase1.py --probe-cmr`; inspect each NISAR collection's "
                             "'associations' for a service entry, then check Harmony's documented "
                             "service list. Until verified, use full-file download or remote HDF5 "
                             "access (earthaccess.open) with decimated reads."}


def probe_cmr():
    return {"harmony": harmony_status(), "collections": cmr_nisar_collections()}
