EARTH PULSE - Phases 1-3
A satellite image shows a moment. Earth Pulse helps us understand the process.

NASA Space Apps 2026 - Dancing with the SARs. API-first: all observations come from the official ASF Search API (asf_search) for NASA-ISRO NISAR GCOV products. No data is bundled, manually prepared or synthetic.

Status
Phase 1-3 code complete. Neither has been run against live NASA services (development sandbox had no network). Phase 2 was started before Phase 1 live verification at the owner's request.

QUICK START - live streaming, nothing saved to your PC
pip install -r requirements.txt
set EARTHDATA_TOKEN=your-token          (Windows cmd; PowerShell: $env:EARTHDATA_TOKEN="your-token")
streamlit run app.py
Open LIVE DATA: enter an area of interest (lon,lat), click 1. Search NASA archive (ASF Search API, results held in memory), pick a stack, click 2. Stream observations from NASA. Each NISAR product is opened remotely over HTTPS (earthaccess, range reads) and only a window around your area is read into memory. Then use WATCH, INVESTIGATE, COMPARE and FORECAST. No CSV, HDF5 or array file is written (tests assert this). Data disappear when the session ends; "refresh" simply repeats the search. The CSV/download CLI below is optional and not used by the app.

Install & authenticate
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export EARTHDATA_TOKEN="..."      # token from https://urs.earthdata.nasa.gov/ (or USERNAME+PASSWORD)
Or copy .streamlit/secrets.toml.example to .streamlit/secrets.toml (git-ignored). Public searching needs no login; downloading does. Credentials are never logged or stored.

Run order (do these on a networked machine)
python -m unittest tests.test_phase1 -v     # offline logic tests (h5py test runs if installed)
python run_phase1.py --probe                # print REAL property keys of a few NISAR results
python run_phase1.py --probe-cmr            # NISAR collections in CMR + Harmony status
python run_phase1.py --refresh --aoi LON,LAT           # live search for ONE place (all dates)
python run_phase1.py --refresh --aoi MINLON,MINLAT,MAXLON,MAXLAT   # or a box (repeat --aoi for several)
python run_phase1.py --refresh --grid-step 30          # global SAMPLE: one point per 30-degree cell
python run_phase1.py --refresh --aoi LON,LAT --download 1   # also download + inspect + read 1 product
streamlit run app.py                        # status page (source, origin, last refresh)
python run_phase1.py --from-cache           # rebuild outputs from last live search (no network)
Outputs (data/metadata/): nisar_search_manifest.{csv,json}, observation_inventory.csv, candidate_stacks.csv, data_quality_report.json, search_summary.txt, search_state.json, search_cache.json, retrieval_log.json.

Method summary
Pagination: each window returning >= 1000 records (or flagged incomplete by the library) is bisected until it fits; unresolvable windows are reported and search complete = False.
Inventory: only .h5 science products; QA/stats/auxiliary files excluded and counted; deduplicated on actual product ID; conflicting duplicates flagged.
Metadata: taken from API fields; missing values stay empty and are listed. File-name parsing is a fallback labelled filename_derived_unverified. Maturity is detected only if a VALIDATED/PROVISIONAL/BETA token appears in API metadata.
Stacks: key = product type, track, direction, frame, polarization, maturity, release (unknown = unstackable). Ranking is lexicographic and documented in modules/observation_ranking.py; the selection reason is generated from computed metrics.
Integrity guard: test fixtures (TESTFIXTURE_ ids) are rejected by the real pipeline.
Verification status (honest)
Item	State
Offline unit tests (26)	25 passed, 1 skipped (HDF5 - h5py absent in sandbox)
Live ASF search, real product IDs, real metadata	NOT RUN
Real HDF5 read of a NISAR file	NOT RUN
Streamlit page rendering	NOT RUN (compiled only)
Harmony NISAR subsetting	UNVERIFIED (probe provided)
Earthdata auth against live service	NOT RUN
Known assumptions to confirm with real output
ASF property names (pathNumber, frameNumber, flightDirection, polarization, processingRelease...) and how maturity/release appear for NISAR - confirm via --probe.
asf_search accepts dataset="NISAR", processingLevel="GCOV", searchComplete.
GCOV HDF5 layout (science/LSAR/GCOV/grids/frequencyA/HHHH...) - confirm via inspect_h5.
Phase 2 - EARTH PULSE Intelligence
python run_phase1.py --refresh                 # archive search (Phase 1)
python build_stack.py --rank 1 --max-products 12   # download + extract best stack -> data/processed/
streamlit run app.py                           # HOME / WATCH / INVESTIGATE / COMPARE / ABOUT
python -m unittest tests.test_phase1 tests.test_phase2 -v
Data path: ranked stack -> build_stack.py downloads each product from ASF (Earthdata auth), reads the chosen diagonal covariance layer (decimated), saves arrays + stack_manifest.json with product ID, URL, retrieval status. The app loads only entries with origin NASA_PRODUCT_FILE and an accepted retrieval status; test fixtures, failed or unknown entries are refused and listed.
No stack cached -> empty state; the app never shows placeholder observations.
Signature: documented threshold rules in modules/pulse_analysis.py (also in the app). The 0.5 dB threshold is an analyst parameter exposed as a slider; no confidence scores are produced.
Compare: blocked unless product type, polarization, maturity, release, layer and units match; geometry (direction/track) differences restrict to relative change; time/space overlap reported.
Evidence panel on INVESTIGATE lists source, product IDs, dates, retrieval status, product type, maturity, release, layer, units, processing and calculation method, limitations.
Phase 2 verification report
Item	State
Offline tests (52)	51 passed, 1 skipped (HDF5, h5py absent)
Signature rules: sudden, persistent, gradual, reversing, repeated, none, insufficient, NaN, threshold	tested (synthetic fixtures, logic only)
Change story from computed values; cause disclaimer; unavailable units	tested
Compare gating (each blocking field, unknown, geometry, no time overlap)	tested
Loader refuses fixtures / failed / non-NASA / missing files	tested
build_stack.py against real products	NOT RUN
Streamlit UI (navigation, plots, WATCH with real data)	NOT RUN (py_compile only)
INVESTIGATE / COMPARE on real NISAR stacks	NOT RUN
Layer choice, units attribute, fill values on real GCOV files	UNVERIFIED
Phase 3 - EARTH PULSE FORECAST
Run: python -m unittest discover -s tests -v, then streamlit run app.py -> FORECAST.

What it forecasts: the mean backscatter (dB over common valid pixels) of a stack's area at the next acquisition, one median historical interval after the last observation (the real date is unknown - stated in the app). It does not predict disasters or imply cause.

Method (modules/forecast.py): one row per real observation (no interpolation); features use only earlier observations (last value, last change, rate, rolling mean/std of 3, real interval, last valid-pixel %); target is the next measured change. Models: persistence baseline, LinearRegression, RandomForest (seed 42). Last ~30% of rows are the held-out test; model selection uses walk-forward validation inside the training rows only, and a model replaces persistence only if it has lower validation RMSE. Test rows are evaluated afterwards by expanding-window walk-forward (MAE, RMSE vs persistence). Uncertainty: empirical residual interval only with >= 10 residuals, otherwise none. Fewer than 10 observations -> "Forecast unavailable: insufficient compatible historical observations." These minimums (10 observations, 6 training rows) are conservative design choices, not statistical guarantees.

Phase 3 verification report

Item	State
Offline tests (71 total)	70 passed, 1 skipped (HDF5 test, h5py absent)
Phase 3 tests (19): unavailable gating, no interpolation, no feature leakage (future values corrupted), walk-forward uses only earlier rows, selection independent of test rows, persistence metrics by hand, trend beats persistence, ties to persistence, reproducibility, interval gating, labelling	passed, synthetic fixtures
Forecast on a real NISAR stack	NOT RUN (no real stack available here)
Whether any real stack has >= 10 observations	UNKNOWN
FORECAST page rendering	NOT RUN (py_compile only)
Real-data model performance	NONE REPORTED - none exists yet
Finding from the first live run (user's log)
The ASF search works. NISAR GCOV records appear from about late Sept/Oct 2025 and number hundreds to over a thousand per day, so listing the whole archive exhausts memory. Searching is therefore by place (--aoi) or by a coarse global sample (--grid-step); a plain --refresh is refused. A grid search is a SAMPLE, not exhaustive, and says so in its summary. Records are slimmed on ingestion (short scalar properties kept verbatim, footprint stored as its bounding box).

Streaming mode (modules/live_stream.py) - verification status
Item	State
Offline tests (97): in-memory discovery, no files written, failures reported, grid-mismatch exclusion, window geometry, latest-N selection, handles closed	96 passed, 1 skipped
Live ASF search by area (intersectsWith)	NOT RUN (global date-range search worked in user's log; AOI search untested)
earthaccess.open remote HDF5 range reads of NISAR GCOV	NOT RUN
Real GCOV layout assumptions: xCoordinates/yCoordinates/projection (EPSG) beside the layer	UNVERIFIED - on mismatch the app shows the group's real dataset names
Whether the API supplies maturity/release (needed for strict stacking)	UNKNOWN - LIVE DATA shows what is missing; a labelled UNVERIFIED option groups without maturity
Streamlit UI rendering	NOT RUN (py_compile only)
Standout extensions (November 2026)
AI Earth Scientist is a deterministic evidence-grounded narrative assistant, not a generative AI model. It uses only already downloaded NISAR observations and exports an auditable JSON report. It never identifies causes from radar backscatter alone.
3D Earth Explorer is a tilted interactive map (not a spherical globe) showing pins for georeferenced, loaded NASA stacks. Basemap tiles are contextual third-party cartography, not NASA data.
Environmental Watchlist stores product-ID snapshots only in the current Streamlit session. To detect new acquisitions, rerun LIVE DATA search/stream and refresh the watchlist. It does not send scheduled notifications.
All original pages and analysis paths are retained. These additions do not synthesize satellite measurements. Install the added pydeck dependency from requirements.txt.
