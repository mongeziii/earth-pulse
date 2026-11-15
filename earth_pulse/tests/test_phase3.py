"""Phase 3 tests. Series are SYNTHETIC TEST FIXTURES for logic only - never shown in the app."""
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import forecast as F

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def mk(values, gaps=None):
    gaps = gaps or [12] * (len(values) - 1)
    t, out = T0, [T0.isoformat()]
    for g in gaps:
        t = t + timedelta(days=g); out.append(t.isoformat())
    return out, list(values), [95.0] * len(values), [f"TESTFIXTURE_{i}" for i in range(len(values))]


def run(values, gaps=None):
    d, v, p, ids = mk(values, gaps)
    return F.run_forecast(d, v, p, ids)


def trend(n, step=-0.5, start=-10.0):
    return [start + step * i for i in range(n)]


class TestAvailability(unittest.TestCase):
    def test_insufficient_history_blocks_forecast(self):
        for n in (1, 3, 9):
            r = run(trend(n))
            self.assertEqual(r["status"], "unavailable")
            self.assertEqual(r["message"], "Forecast unavailable: insufficient compatible historical observations.")
            self.assertNotIn("forecast", r)

    def test_minimum_available(self):
        self.assertEqual(run(trend(10))["status"], "ok")

    def test_explanation_for_unavailable(self):
        self.assertIn("Forecast unavailable", F.forecast_explanation(run(trend(4))))


class TestDataset(unittest.TestCase):
    def test_rows_equal_real_observations_minus_one_no_interpolation(self):
        d, v, p, _ = mk(trend(12), gaps=[12, 12, 24, 12, 36, 12, 12, 12, 12, 12, 12])
        X, y, last, idx = F.build_dataset(d, v, p)
        self.assertEqual(len(y), 11)
        gi = F.FEATURES.index("interval_days")
        self.assertEqual(X[2, gi], 24.0)           # actual gap, not a regular grid
        self.assertEqual(X[4, gi], 36.0)

    def test_no_future_leakage_in_features(self):
        v = list(np.random.default_rng(1).normal(-10, 1, 16))
        d, _, p, _ = mk(v)
        X1, y1, *_ = F.build_dataset(d, v, p)
        m = 8
        v2 = v[:m] + [x + 50 for x in v[m:]]       # corrupt observations m.. onward
        X2, y2, *_ = F.build_dataset(d, v2, p)
        np.testing.assert_array_equal(X1[:m], X2[:m])   # rows whose target index <= m: features unchanged
        np.testing.assert_array_equal(y1[:m - 1], y2[:m - 1])

    def test_rejects_bad_inputs(self):
        d, v, p, _ = mk(trend(12))
        with self.assertRaises(ValueError): F.build_dataset(d[::-1], v, p)        # unordered
        with self.assertRaises(ValueError): F.build_dataset(d[:-1], v, p)         # length mismatch
        with self.assertRaises(ValueError): F.build_dataset(["bad"] + d[1:], v, p)


class TestChronology(unittest.TestCase):
    def test_walk_forward_trains_only_on_earlier_rows(self):
        X = np.random.default_rng(0).normal(size=(12, 7)); y = np.random.default_rng(1).normal(size=12)
        _, sizes = F.walk_forward("linear_regression", X, y, list(range(4, 12)))
        self.assertEqual(sizes, list(range(4, 12)))   # row i predicted from rows[:i]

    def test_test_period_after_training_period(self):
        r = run(list(np.random.default_rng(3).normal(-10, 1, 20)))
        self.assertLess(r["training_period"][1], r["test_period"][0])
        self.assertEqual(r["n_train_rows"] + r["n_test_rows"], r["n_rows"])

    def test_selection_ignores_test_rows(self):
        v = list(np.random.default_rng(5).normal(-10, 1, 20))
        a = run(v)
        cut = a["n_train_rows"] + 1                  # observations beyond this feed only test rows
        v2 = v[:cut] + [x + 30 for x in v[cut:]]
        b = run(v2)
        self.assertEqual(a["validation_metrics"], b["validation_metrics"])
        self.assertEqual(a["selected_model"], b["selected_model"])
        self.assertNotEqual(a["test_metrics"], b["test_metrics"])


class TestModels(unittest.TestCase):
    def test_persistence_metrics_by_hand(self):
        r = run(trend(14, step=-0.5))
        p = r["test_metrics"]["persistence"]
        self.assertAlmostEqual(p["mae"], 0.5); self.assertAlmostEqual(p["rmse"], 0.5)

    def test_trend_beats_persistence(self):
        r = run(trend(16, step=-0.5))
        self.assertNotEqual(r["selected_model"], "persistence")
        self.assertTrue(r["beats_persistence_on_test"][r["selected_model"]])
        self.assertAlmostEqual(r["forecast"]["predicted_change_db"], -0.5, places=1)

    def test_ties_go_to_persistence(self):
        r = run([-10.0] * 14)
        self.assertEqual(r["selected_model"], "persistence")
        self.assertAlmostEqual(r["forecast"]["value_db"], -10.0)

    def test_all_three_models_reported(self):
        r = run(list(np.random.default_rng(2).normal(-10, 1, 16)))
        self.assertEqual(set(r["test_metrics"]), {"persistence", "linear_regression", "random_forest"})
        self.assertEqual(set(r["validation_metrics"]), set(r["test_metrics"]))

    def test_forecast_value_is_last_plus_change(self):
        r = run(list(np.random.default_rng(4).normal(-10, 1, 16)))
        f = r["forecast"]
        self.assertAlmostEqual(f["value_db"], r["last_observation"]["value_db"] + f["predicted_change_db"])

    def test_reproducible(self):
        v = list(np.random.default_rng(7).normal(-10, 1, 18))
        self.assertEqual(json.dumps(run(v), sort_keys=True), json.dumps(run(v), sort_keys=True))

    def test_fingerprint_changes_with_data(self):
        v = list(np.random.default_rng(7).normal(-10, 1, 14))
        w = v[:-1] + [v[-1] + 1]
        self.assertNotEqual(run(v)["data_fingerprint"], run(w)["data_fingerprint"])


class TestForecastOutput(unittest.TestCase):
    def test_horizon_uses_median_real_gap(self):
        gaps = [12, 12, 12, 24, 12, 12, 12, 12, 36, 12, 12]
        r = run(list(np.random.default_rng(6).normal(-10, 1, 12)), gaps)
        self.assertEqual(r["forecast"]["horizon_days"], 12.0)
        last = datetime.fromisoformat(r["last_observation"]["date"])
        self.assertEqual(r["forecast"]["forecast_date"], (last + timedelta(days=12)).date().isoformat())
        self.assertIn("not known", r["forecast"]["horizon_assumption"])

    def test_interval_only_with_enough_residuals(self):
        small = run(list(np.random.default_rng(8).normal(-10, 1, 12)))
        self.assertIsNone(small["forecast"]["interval"]); self.assertIn("No uncertainty interval", small["forecast"]["interval_note"])
        big = run(list(np.random.default_rng(8).normal(-10, 1, 24)))
        iv = big["forecast"]["interval"]
        self.assertIsNotNone(iv); self.assertLess(iv["lower"], big["forecast"]["value_db"] < iv["upper"])
        self.assertGreaterEqual(iv["n_residuals"], F.MIN_RESIDUALS_FOR_INTERVAL)

    def test_labelled_as_estimate_and_limitations(self):
        r = run(trend(14))
        self.assertIn("not a NASA observation", r["label"]); self.assertTrue(r["limitations"])
        t = F.forecast_explanation(r)
        for s in ("model estimate, not a NASA observation", "persistence", "No cause is implied", "could be wrong"):
            self.assertIn(s, t)
        self.assertIn(f"{r['n_observations']} real NISAR observations", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
