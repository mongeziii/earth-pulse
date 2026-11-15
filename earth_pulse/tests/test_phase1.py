"""Phase 1 tests. All records here are SYNTHETIC TEST FIXTURES (ids prefixed TESTFIXTURE_).
They exercise logic only; they are never real NASA data and never written to data/."""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import nasa_api
from modules.data_validation import check_date_order, compatibility, validate_array
from modules.earthdata_auth import AuthError, build_asf_session, redact
from modules.nisar_search import search_archive, search_interval
from modules.observation_ranking import bbox_overlap, build_stacks, explain_selection
from modules.product_metadata import dedupe, is_qa_or_auxiliary, normalize_record

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fx(i, day=0, track=1, frame=2, pol="HH+HV", mat="PROVISIONAL", rel="X00001",
       name=None, bbox=(10, 10, 12, 12), drop=()):
    start = (T0 + timedelta(days=day)).strftime("%Y-%m-%dT%H:%M:%SZ")
    nm = name or f"TESTFIXTURE_{i}.h5"
    props = {"sceneName": nm[:-3], "fileName": nm, "startTime": start, "processingLevel": "GCOV",
             "pathNumber": track, "frameNumber": frame, "flightDirection": "ASCENDING",
             "polarization": pol, "collectionName": f"FAKE_{mat}", "processingRelease": rel,
             "url": "https://example.invalid/" + nm}
    for d in drop:
        props.pop(d, None)
    x0, y0, x1, y1 = bbox
    return {"properties": props,
            "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}


class TestPagination(unittest.TestCase):
    def make_archive(self, n):
        times = [T0 + timedelta(hours=i) for i in range(n)]
        calls = []

        def search_fn(s, e, cap):
            calls.append((s, e))
            hits = [fx(i) | {"t": t} for i, t in enumerate(times) if s <= t < e]
            return hits[:cap], None
        return times, search_fn, calls

    def test_truncation_is_bisected_and_complete(self):
        n = 50
        _, fn, calls = self.make_archive(n)
        recs, rep = search_interval(fn, T0, T0 + timedelta(hours=n), cap=10, min_span_hours=1)
        self.assertEqual(len(recs), n)                 # nothing lost to the cap
        self.assertTrue(rep["complete"])
        self.assertGreater(len(calls), 5)              # it actually split

    def test_unresolvable_truncation_is_reported(self):
        def fn(s, e, cap):                              # always full -> can never fit
            return [fx(i) for i in range(cap)], None
        _, rep = search_interval(fn, T0, T0 + timedelta(hours=8), cap=5, min_span_hours=4)
        self.assertFalse(rep["complete"])
        self.assertTrue(rep["unresolved_truncated_windows"])

    def test_api_incomplete_flag_triggers_split(self):
        seen = []
        def fn(s, e, cap):
            seen.append((e - s))
            return [], (False if (e - s) > timedelta(hours=12) else True)
        _, rep = search_interval(fn, T0, T0 + timedelta(days=2), cap=100, min_span_hours=1)
        self.assertTrue(rep["complete"]); self.assertGreater(len(seen), 1)

    def test_chunked_search_aggregates(self):
        _, fn, _ = self.make_archive(100)
        recs, rep = search_archive(fn, T0, T0 + timedelta(hours=100), chunk_days=1, cap=1000)
        self.assertEqual(len(recs), 100); self.assertTrue(rep["complete"])


class TestBudgetAndProgress(unittest.TestCase):
    def test_call_budget_stops_runaway_and_reports_incomplete(self):
        calls = []
        def fn(s, e, cap):                      # archive that always returns a full page
            calls.append(1); return [fx(i) for i in range(cap)], None
        _, rep = search_archive(fn, T0, T0 + timedelta(days=400), chunk_days=30, cap=5, max_calls=25)
        self.assertLessEqual(len(calls), 25); self.assertFalse(rep["complete"])
        self.assertTrue(rep["budget_exhausted"]); self.assertEqual(rep["calls_used"], 25)

    def test_progress_callback_called_per_request(self):
        seen = []
        search_archive(lambda s, e, cap: ([], True), T0, T0 + timedelta(days=65), 30, 10, 50, seen.append)
        self.assertEqual(len(seen), 3); self.assertEqual(seen[-1]["call"], 3)


class TestSlimAndAoi(unittest.TestCase):
    def test_slim_record_keeps_scalars_drops_bulk_and_boxes_geometry(self):
        props = {"fileName": "a.h5", "pathNumber": 7, "ok": True, "huge": "x" * 5000, "nested": {"a": 1}, "none": None}
        geom = {"type": "Polygon", "coordinates": [[[0, 0], [4, 1], [3, 5], [-1, 3], [0, 0]]]}
        r = nasa_api.slim_record(props, geom)
        self.assertEqual(set(r["properties"]), {"fileName", "pathNumber", "ok", "none"})
        from modules.product_metadata import footprint_bbox
        self.assertEqual(footprint_bbox(r["geometry"]), (-1.0, 0.0, 4.0, 5.0))

    def test_slim_record_without_geometry(self):
        self.assertIsNone(nasa_api.slim_record({"a": 1}, None)["geometry"])

    def test_parse_aoi(self):
        from modules.aoi import parse_aoi
        self.assertEqual(parse_aoi("31.0,-29.8")[1], "POINT(31.0 -29.8)")
        self.assertTrue(parse_aoi("10,10,12,12")[1].startswith("POLYGON(("))
        for bad in ("abc", "1", "200,0", "0,100", "5,5,4,4", "1,2,3"):
            with self.assertRaises(ValueError): parse_aoi(bad)

    def test_grid_points(self):
        from modules.aoi import grid_points
        g = grid_points(30)
        self.assertEqual(len(g), 72)
        self.assertTrue(all(-90 < la < 90 and -180 < lo < 180 for la, lo in g))
        with self.assertRaises(ValueError): grid_points(1)

    def test_cli_refuses_blind_archive_listing(self):
        import subprocess
        r = subprocess.run([sys.executable, "run_phase1.py", "--refresh"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parent.parent)
        self.assertEqual(r.returncode, 5); self.assertIn("Choose where to search", r.stderr)


class TestMetadata(unittest.TestCase):
    def test_normalize_and_missing_fields(self):
        o = normalize_record(fx(1))
        self.assertEqual(o["maturity"], "PROVISIONAL"); self.assertEqual(o["missing_fields"], [])
        self.assertEqual(o["acquisition_date"], "2026-01-01")
        self.assertTrue(o["is_test_fixture"])

    def test_missing_metadata_is_none_not_invented(self):
        o = normalize_record(fx(2, drop=("pathNumber", "polarization", "processingRelease", "collectionName")))
        for f in ("track", "polarization", "release", "maturity"):
            self.assertIsNone(o[f]); self.assertIn(f, o["missing_fields"])
        self.assertEqual(o["field_sources"]["polarization"], "unavailable")

    def test_filename_fallback_is_labelled_unverified(self):
        name = "NISAR_L2_PR_GCOV_004_077_A_123_4005_SHNA_A_20260105T010203_20260105T010230_P05007_F_N_J_001.h5"
        o = normalize_record({"properties": {"fileName": name}, "geometry": None})
        self.assertEqual((o["track"], o["frame"], o["release"]), (77, 123, "P05007"))
        self.assertEqual(o["field_sources"]["track"], "filename_derived_unverified")

    def test_qa_excluded(self):
        qa = normalize_record(fx(3, name="TESTFIXTURE_NISAR_GCOV_QA_STATS.h5"))
        png = normalize_record(fx(4, name="TESTFIXTURE_browse.png"))
        ok = normalize_record(fx(5))
        self.assertTrue(is_qa_or_auxiliary(qa)); self.assertTrue(is_qa_or_auxiliary(png))
        self.assertFalse(is_qa_or_auxiliary(ok))

    def test_dedupe_uses_product_id(self):
        a, b, c = normalize_record(fx(1)), normalize_record(fx(1)), normalize_record(fx(2))
        u, rep = dedupe([a, b, c])
        self.assertEqual(len(u), 2); self.assertEqual(rep["duplicates_removed"], 1)

    def test_dedupe_flags_conflicts_and_idless(self):
        a, b = normalize_record(fx(1, day=0)), normalize_record(fx(1, day=5))
        nid = normalize_record({"properties": {}, "geometry": None})
        u, rep = dedupe([a, b, nid])
        self.assertEqual(rep["conflicting_duplicates"], ["TESTFIXTURE_1"])
        self.assertEqual(rep["records_without_product_id"], 1)


class TestValidation(unittest.TestCase):
    def test_valid_array(self):
        r = validate_array(np.array([[1.0, 2.0], [3.0, np.nan]]))
        self.assertTrue(r["ok"]); self.assertEqual(r["valid_pixel_pct"], 75.0)

    def test_invalid_arrays(self):
        self.assertFalse(validate_array(np.array([]))["ok"])
        self.assertFalse(validate_array(np.full((3, 3), np.nan))["ok"])
        self.assertFalse(validate_array(np.array([[1.0, -2.0]]))["ok"])
        self.assertFalse(validate_array(np.ones((3, 3)))["ok"])      # constant
        self.assertEqual(validate_array(np.zeros((2, 2)))["usable_pixels"], 0)

    def test_fill_value_masked(self):
        r = validate_array(np.array([[-9999.0, 1.0, 2.0]]), fill_value=-9999.0)
        self.assertEqual(r["negative_pixels"], 0); self.assertAlmostEqual(r["missing_pct"], 100 / 3, 2)

    def test_date_order(self):
        obs = [normalize_record(fx(i, day=d)) for i, d in ((1, 9), (2, 1), (3, 5))]
        obs.append(normalize_record({"properties": {"sceneName": "TESTFIXTURE_nodate"}, "geometry": None}))
        s, rep = check_date_order(obs)
        self.assertEqual([o["product_id"] for o in s], ["TESTFIXTURE_2", "TESTFIXTURE_3", "TESTFIXTURE_1"])
        self.assertEqual(rep["undated_product_ids"], ["TESTFIXTURE_nodate"])

    def test_compatibility(self):
        a, b = normalize_record(fx(1)), normalize_record(fx(2))
        self.assertTrue(compatibility(a, b)[0])
        self.assertFalse(compatibility(a, normalize_record(fx(3, pol="HH")))[0])
        self.assertFalse(compatibility(a, normalize_record(fx(4, mat="VALIDATED")))[0])
        ok, why = compatibility(a, normalize_record(fx(5, drop=("processingRelease",))))
        self.assertFalse(ok); self.assertTrue(any("release" in w for w in why))


class TestRanking(unittest.TestCase):
    def test_bbox_overlap(self):
        self.assertEqual(bbox_overlap((0, 0, 2, 2), (0, 0, 2, 2)), 1.0)
        self.assertEqual(bbox_overlap((0, 0, 1, 1), (5, 5, 6, 6)), 0.0)
        self.assertAlmostEqual(bbox_overlap((0, 0, 2, 2), (1, 0, 3, 2)), 0.5)

    def test_stacks_rank_and_eligibility(self):
        obs = [normalize_record(fx(i, day=12 * i, track=1)) for i in range(4)]       # 4-obs stack
        obs += [normalize_record(fx(10 + i, day=6 * i, track=2)) for i in range(2)]  # too small
        obs += [normalize_record(fx(20, track=3, drop=("polarization",)))]           # unstackable
        rows, info = build_stacks(obs)
        self.assertEqual(rows[0]["track"], 1); self.assertTrue(rows[0]["eligible"])
        self.assertFalse(rows[1]["eligible"]); self.assertEqual(info["unstackable_count"], 1)
        self.assertEqual(rows[0]["max_gap_days"], 12.0)
        self.assertIn("4 distinct acquisitions", explain_selection(rows))

    def test_same_timestamp_counts_once(self):
        obs = [normalize_record(fx(i, day=0)) for i in range(3)]
        rows, _ = build_stacks(obs)
        self.assertEqual(rows[0]["n_observations"], 1); self.assertFalse(rows[0]["eligible"])

    def test_no_eligible_stack_is_honest(self):
        rows, _ = build_stacks([normalize_record(fx(1)), normalize_record(fx(2, day=3))])
        self.assertIn("NO stack meets", explain_selection(rows))

    def test_maturity_not_mixed(self):
        obs = [normalize_record(fx(i, day=i * 5, mat="PROVISIONAL" if i % 2 else "VALIDATED")) for i in range(4)]
        rows, _ = build_stacks(obs)
        self.assertEqual(len(rows), 2); self.assertTrue(all(not r["eligible"] for r in rows))

    def test_empty(self):
        self.assertIn("No stacks", explain_selection(build_stacks([])[0]))


class TestAuthAndAccess(unittest.TestCase):
    def test_redact(self):
        self.assertNotIn("s3cret", redact("bad token s3cret", ["s3cret"]))

    def test_missing_credentials_message(self):
        for k in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
            os.environ.pop(k, None)
        with self.assertRaises(AuthError) as cm:
            build_asf_session()
        self.assertIn("EARTHDATA_TOKEN", str(cm.exception))

    def test_archive_failure_is_explicit_not_faked(self):
        try:
            import asf_search  # noqa: F401
            self.skipTest("asf_search installed; failure path covered by network-dependent run")
        except ImportError:
            with self.assertRaises(nasa_api.ArchiveAccessError):
                nasa_api.raw_search(T0, T0 + timedelta(days=1), 10)

    def test_pipeline_refuses_fixtures(self):
        import run_phase1
        with self.assertRaises(RuntimeError):
            run_phase1.build_outputs([fx(1)], {"raw_record_count": 1, "n_windows": 1, "complete": True},
                                     "LIVE_API", "x")


try:
    import h5py
except ImportError:
    h5py = None


@unittest.skipIf(h5py is None, "h5py not installed in this environment - HDF5 tests NOT run")
class TestHDF5(unittest.TestCase):
    """Uses a tiny locally generated HDF5 file (a TEST FIXTURE, not NISAR data)."""
    def test_layer_discovery_and_read(self):
        from modules.data_extraction import find_gcov_layers, inspect_h5, read_layer
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "TESTFIXTURE.h5"
            with h5py.File(p, "w") as f:
                g = f.create_group("science/LSAR/GCOV/grids/frequencyA")
                ds = g.create_dataset("HHHH", data=np.random.rand(32, 32).astype("float32") + 0.1)
                ds.attrs["units"] = "1"; ds.attrs["_FillValue"] = np.float32(-9999)
                g.create_dataset("HHHV", data=np.ones((4, 4), dtype="complex64"))
            layers = find_gcov_layers(p)
            self.assertEqual({l["polarization_term"] for l in layers}, {"HHHH"})
            self.assertTrue(inspect_h5(p))
            r = read_layer(p, layers[0]["path"], stride=2)
            self.assertEqual(r["array"].shape, (16, 16)); self.assertTrue(r["quality"]["ok"])
            with self.assertRaises(ValueError):
                read_layer(p, "science/LSAR/GCOV/grids/frequencyA/HHHV")


if __name__ == "__main__":
    unittest.main(verbosity=2)
