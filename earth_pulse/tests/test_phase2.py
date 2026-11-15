"""Phase 2 tests. Arrays are SYNTHETIC TEST FIXTURES for logic only - never shown in the app."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.pulse_analysis import change_story, classify_signature, evidence_panel, series_from_arrays, to_db
from modules.pulse_compare import compare_stacks
from modules.stack_store import load_stack, slug


def arrs(db_means, shape=(20, 20), seed=0):
    """Linear-power arrays whose common-mask mean dB equals db_means (constant offset on a fixed texture)."""
    tex = np.random.default_rng(seed).normal(0, 1, shape)
    tex -= tex.mean()
    return [10 ** ((m + tex) / 10) for m in db_means]


def label(means, **kw):
    s = series_from_arrays(arrs(means))
    return classify_signature(s["mean_db"], s["common_valid_pct"], **kw)


class TestSeries(unittest.TestCase):
    def test_means_recovered(self):
        s = series_from_arrays(arrs([-10, -12, -11]))
        np.testing.assert_allclose(s["mean_db"], [-10, -12, -11], atol=1e-9)
        self.assertEqual(s["common_valid_pct"], 100.0)

    def test_common_mask_excludes_invalid_in_any_date(self):
        a = arrs([-10, -10, -10]); a[1][0, :] = np.nan; a[2][:, 0] = 0.0
        s = series_from_arrays(a)
        self.assertLess(s["common_valid_pct"], 100.0)
        self.assertEqual(s["per_obs_valid_pct"][0], 100.0)

    def test_shape_mismatch_blocked(self):
        with self.assertRaises(ValueError):
            series_from_arrays([np.ones((4, 4)), np.ones((5, 5))])

    def test_to_db_masks_nonpositive(self):
        r = to_db(np.array([1.0, 0.0, -1.0, np.nan, 10.0]))
        self.assertTrue(np.isnan(r[1:4]).all()); self.assertEqual(r[4], 10.0)


class TestSignature(unittest.TestCase):
    def test_sudden_persistent(self):
        r = label([-10, -10.1, -13, -13.1, -13.0])
        self.assertEqual(r["label"], "SUDDEN CHANGE"); self.assertIn("PERSISTENT", r["tags"])

    def test_sudden_not_persistent_when_no_followup(self):
        r = label([-10, -10.1, -13])
        self.assertEqual(r["label"], "SUDDEN CHANGE"); self.assertEqual(r["tags"], [])

    def test_gradual(self):
        self.assertEqual(label([-10, -10.7, -11.4, -12.1, -12.8])["label"], "GRADUAL CHANGE")

    def test_reversing(self):
        self.assertEqual(label([-10, -13, -10.1])["label"], "REVERSING PATTERN")

    def test_repeated(self):
        self.assertEqual(label([-10, -12, -10, -12, -10])["label"], "REPEATED VARIATION")

    def test_no_change(self):
        self.assertEqual(label([-10, -10.1, -9.95, -10.05])["label"], "NO DETECTABLE CHANGE")

    def test_insufficient_observations(self):
        self.assertEqual(label([-10, -12])["label"], "INSUFFICIENT EVIDENCE")

    def test_insufficient_valid_pixels(self):
        r = classify_signature([-10, -12, -14], 3.0)
        self.assertEqual(r["label"], "INSUFFICIENT EVIDENCE")

    def test_nan_series_insufficient(self):
        self.assertEqual(classify_signature([-10, float("nan"), -12], 100)["label"], "INSUFFICIENT EVIDENCE")

    def test_threshold_changes_outcome(self):
        means = [-10, -10.4, -10.8]
        self.assertEqual(label(means, threshold_db=1.0)["label"], "NO DETECTABLE CHANGE")
        self.assertEqual(label(means, threshold_db=0.3)["label"], "GRADUAL CHANGE")

    def test_no_confidence_score_emitted(self):
        r = label([-10, -13, -13.1])
        self.assertNotIn("confidence", json.dumps(r, default=str).lower())


class TestStory(unittest.TestCase):
    def test_story_uses_computed_values_and_denies_cause(self):
        means = [-10, -10.1, -13, -13.1]
        s = series_from_arrays(arrs(means)); sig = classify_signature(s["mean_db"], s["common_valid_pct"])
        t = change_story(["2026-01-01", "2026-01-13", "2026-01-25", "2026-02-06"], sig, s, "1")
        self.assertIn("-3.10 dB", t); self.assertIn("4 valid observations", t)
        self.assertIn("do not establish the underlying cause", t); self.assertIn("2026-01-01", t)

    def test_story_insufficient_states_unavailable(self):
        sig = classify_signature([-10, -11], 100.0)
        t = change_story(["a", "b"], sig, {"common_valid_pct": 100}, None)
        self.assertIn("insufficient", t.lower()); self.assertNotIn("changed by", t)

    def test_unknown_units_stated(self):
        s = series_from_arrays(arrs([-10, -12, -14])); sig = classify_signature(s["mean_db"], 100.0)
        self.assertIn("unavailable", change_story(["a", "b", "c"], sig, s, None))


class TestEvidence(unittest.TestCase):
    def test_panel_has_required_fields(self):
        e = [{"product_id": "P1", "start_time": "t1", "retrieval_status": "DOWNLOADED_VERIFIED"}]
        p = evidence_panel(e, {"product_type": "GCOV", "maturity": "PROVISIONAL", "release": "X"}, "a/HHHH", "1", 8)
        for k in ("source", "product_ids", "acquisition_dates", "product_type", "data_maturity", "processing_release",
                  "measurement_layer", "units", "processing_method", "calculation_method", "limitations"):
            self.assertIn(k, p)


BASE = dict(product_type="GCOV", polarization="HH+HV", maturity="PROVISIONAL", release="R1", direction="ASCENDING",
            track=1, layer_path_leaf="HHHH", units="1", bbox=[0, 0, 1, 1],
            first_acquisition="2026-01-01T00:00:00+00:00", last_acquisition="2026-03-01T00:00:00+00:00")


class TestCompare(unittest.TestCase):
    def test_compatible(self):
        r = compare_stacks(BASE, {**BASE, "bbox": [5, 5, 6, 6], "track": 1})
        self.assertTrue(r["allowed"]); self.assertEqual(r["spatial_overlap"], 0.0)

    def test_blocks_on_each_field(self):
        for f, v in (("product_type", "GUNW"), ("polarization", "HH"), ("maturity", "VALIDATED"),
                     ("release", "R2"), ("layer_path_leaf", "HVHV"), ("units", "dB")):
            r = compare_stacks(BASE, {**BASE, f: v})
            self.assertFalse(r["allowed"], f); self.assertIsNone(r["mode"])

    def test_unknown_blocks(self):
        self.assertFalse(compare_stacks(BASE, {**BASE, "units": None})["allowed"])

    def test_geometry_difference_restricts_to_relative(self):
        r = compare_stacks(BASE, {**BASE, "direction": "DESCENDING"})
        self.assertTrue(r["allowed"]); self.assertTrue(r["mode"].startswith("relative")); self.assertTrue(r["warnings"])

    def test_no_temporal_overlap_warns(self):
        r = compare_stacks(BASE, {**BASE, "first_acquisition": "2027-01-01T00:00:00+00:00",
                                  "last_acquisition": "2027-02-01T00:00:00+00:00"})
        self.assertEqual(r["temporal_overlap_days"], 0.0); self.assertTrue(any("time" in w for w in r["warnings"]))


class TestLoader(unittest.TestCase):
    def test_loader_refuses_bad_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"; sd.mkdir()
            for n in ("good", "fixture", "failed", "other"):
                np.save(sd / f"{n}.npy", np.ones((3, 3), "float32"))
            def ent(pid, **kw):
                base = {"product_id": pid, "start_time": "t", "retrieval_status": "DOWNLOADED_VERIFIED",
                        "origin": "NASA_PRODUCT_FILE", "is_test_fixture": False, "array_file": f"{pid}.npy"}
                return {**base, **kw}
            m = {"stack_id": "s", "stack": {}, "entries": [
                ent("good"), ent("fixture", is_test_fixture=True), ent("failed", retrieval_status="FAILED: x"),
                ent("other", origin="manual_upload"), ent("missing", array_file="nope.npy")]}
            (sd / "stack_manifest.json").write_text(json.dumps(m))
            _, ok, arrays, skipped = load_stack("s", root=d)
            self.assertEqual([e["product_id"] for e in ok], ["good"]); self.assertEqual(len(arrays), 1)
            self.assertEqual({s["product_id"] for s in skipped}, {"fixture", "failed", "other", "missing"})

    def test_slug_safe(self):
        self.assertNotIn("|", slug("GCOV|1|ASCENDING|2")) ; self.assertLessEqual(len(slug("x" * 500)), 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
