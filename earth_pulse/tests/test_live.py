"""Live-streaming logic tests. Openers/readers/search functions are SYNTHETIC TEST FIXTURES that replace
NASA network calls; they verify logic only. No NASA service is contacted."""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_phase1 import fx, T0
from modules.earthdata_auth import AuthError, earthaccess_login
from modules.live_stream import StreamError, discover, stream_stack, window_indices
from modules.observation_ranking import UNKNOWN_TEXT, build_stacks
from modules.product_metadata import normalize_record
from modules.pulse_compare import compare_stacks


def snapshot():
    return sorted(str(p) for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in str(p))


def search_for(records):
    return lambda s, e, cap: ([r for r in records if s <= datetime.fromisoformat(r["properties"]["startTime"].replace("Z", "+00:00")) < e + timedelta(seconds=1)], True)


class TestDiscover(unittest.TestCase):
    def setUp(self):
        self.recs = [fx(i, day=12 * i) for i in range(5)]

    def test_in_memory_and_no_files_written(self):
        before = snapshot()
        d = discover("31.0,-29.8", T0 - timedelta(days=1), T0 + timedelta(days=100), search_fn=search_for(self.recs),
                     _allow_test_fixtures=True)
        self.assertEqual(snapshot(), before)                      # nothing written anywhere
        self.assertEqual(len(d["observations"]), 5); self.assertTrue(d["stacks"][0]["eligible"])
        self.assertIn("memory only", d["origin"])

    def test_refuses_fixtures_by_default(self):
        with self.assertRaises(RuntimeError):
            discover("1,1", T0, T0 + timedelta(days=60), search_fn=search_for(self.recs))

    def test_discovery_remembers_searched_aoi(self):
        d = discover(" -119.8,36.7 ", T0 - timedelta(days=1), T0 + timedelta(days=60), search_fn=search_for(self.recs),
                     _allow_test_fixtures=True)
        self.assertEqual(d["aoi_text"], "-119.8,36.7")

    def test_center_lonlat(self):
        from modules.aoi import center_lonlat
        self.assertEqual(center_lonlat("10,20"), (10.0, 20.0))
        self.assertEqual(center_lonlat("0,0,10,20"), (5.0, 10.0))
        with self.assertRaises(ValueError): center_lonlat("")

    def test_bad_aoi_rejected(self):
        with self.assertRaises(ValueError):
            discover("abc", T0, T0 + timedelta(days=5), search_fn=search_for([]))

    def test_missing_maturity_makes_records_unstackable_unless_explicitly_ignored(self):
        recs = [fx(i, day=12 * i, drop=("collectionName",)) for i in range(4)]
        strict = discover("1,1", T0 - timedelta(days=1), T0 + timedelta(days=60), search_fn=search_for(recs), _allow_test_fixtures=True)
        self.assertEqual(strict["stacks"], []); self.assertEqual(strict["unstackable"], 4)
        self.assertEqual(strict["missing_field_counts"]["maturity"], 4)
        loose = discover("1,1", T0 - timedelta(days=1), T0 + timedelta(days=60), ignore_fields=("maturity",),
                         search_fn=search_for(recs), _allow_test_fixtures=True)
        self.assertEqual(loose["stacks"][0]["maturity"], UNKNOWN_TEXT)
        self.assertEqual(loose["stacks"][0]["unverified_fields"], "maturity")

    def test_polarization_empty_for_all_is_auto_ignored_and_labelled(self):
        recs = [fx(i, day=12 * i, drop=("polarization",)) for i in range(4)]
        d = discover("1,1", T0 - timedelta(days=1), T0 + timedelta(days=60), search_fn=search_for(recs), _allow_test_fixtures=True)
        self.assertEqual(d["auto_ignored_fields"], ["polarization"])
        self.assertEqual(len(d["stacks"]), 1); self.assertEqual(d["stacks"][0]["polarization"], UNKNOWN_TEXT)
        self.assertIn("polarization", d["stacks"][0]["unverified_fields"])

    def test_polarization_not_ignored_when_only_some_missing(self):
        recs = [fx(i, day=12 * i, drop=("polarization",) if i == 0 else ()) for i in range(4)]
        d = discover("1,1", T0 - timedelta(days=1), T0 + timedelta(days=60), search_fn=search_for(recs), _allow_test_fixtures=True)
        self.assertEqual(d["auto_ignored_fields"], [])
        self.assertEqual(d["unstackable"], 1)

    def test_stream_sets_polarization_from_file(self):
        recs = [fx(i, day=12 * i, drop=("polarization",)) for i in range(4)]
        d = discover("1,1", T0 - timedelta(days=1), T0 + timedelta(days=60), search_fn=search_for(recs), _allow_test_fixtures=True)
        order = {o["url"]: i for i, o in enumerate(sorted(d["observations"], key=lambda o: o["start_time"]))}
        m, ok, arrays, sk = stream_stack(d["stacks"][0], d["observations"], 1, 1, opener=lambda u: Handle(i=order[u]),
                                         reader=good_reader())
        self.assertEqual(m["stack"]["polarization"], "HHHH (read from files)")
        self.assertNotIn("polarization", m["stack"]["unverified_fields"])

    def test_empty_archive_result_is_honest(self):
        d = discover("1,1", T0, T0 + timedelta(days=5), search_fn=lambda s, e, c: ([], True))
        self.assertEqual(d["observations"], []); self.assertIn("No stacks", d["selection_reason"])


def make_world(n=6, shape=(40, 40)):
    obs = [normalize_record(fx(i, day=12 * i)) for i in range(n)]
    stacks, _ = build_stacks(obs)
    return obs, stacks[0]


def good_reader(grid_shift=None, bad=None):
    def reader(f, lon, lat, half, term, stride):
        i = f["i"]
        if bad and i in bad:
            raise StreamError(bad[i])
        arr = 10 ** ((-10 - i + np.random.default_rng(0).normal(0, 1, (40, 40))) / 10)
        g = {"epsg": 32636, "x_first": 1000.0 + (5 if grid_shift == i else 0), "y_first": 2000.0, "dx": 20.0,
             "dy": -20.0, "shape": [40, 40], "stride": stride}
        q = {"ok": True, "problems": [], "valid_pixel_pct": 100.0}
        return {"array": arr, "units": "1", "layer_path": "g/HHHH", "layer_term": term or "HHHH", "quality": q, "grid": g}
    return reader


class Handle(dict):
    closed = False
    def close(self): self.closed = True


class TestStream(unittest.TestCase):
    def run_stream(self, reader, n=6, **kw):
        obs, row = make_world(n)
        opened, order = [], {o["url"]: i for i, o in enumerate(sorted(obs, key=lambda o: o["start_time"]))}
        def opener(url):
            h = Handle(i=order[url]); opened.append(h); return h
        before = snapshot()
        res = stream_stack(row, obs, 31.0, -29.8, opener=opener, reader=reader, **kw)
        self.assertEqual(snapshot(), before)                       # nothing written to disk
        self.assertTrue(all(h.closed for h in opened))             # every remote handle closed
        return res

    def test_streams_all_in_order_and_labels_status(self):
        m, ok, arrays, skipped = self.run_stream(good_reader())
        self.assertEqual(len(ok), 6); self.assertEqual(len(arrays), 6); self.assertEqual(skipped, [])
        self.assertEqual([e["start_time"] for e in ok], sorted(e["start_time"] for e in ok))
        self.assertTrue(all(e["retrieval_status"] == "STREAMED_NOT_STORED" and e["origin"] == "NASA_PRODUCT_STREAM" for e in ok))
        self.assertIn("no file saved", m["stack"]["window_description"])

    def test_failures_are_reported_not_hidden(self):
        m, ok, arrays, skipped = self.run_stream(good_reader(bad={2: "boom"}))
        self.assertEqual(len(ok), 5); self.assertEqual(len(skipped), 1); self.assertIn("boom", skipped[0]["reason"])

    def test_grid_mismatch_excluded(self):
        m, ok, arrays, skipped = self.run_stream(good_reader(grid_shift=3))
        self.assertEqual(len(ok), 5); self.assertIn("grid differs", skipped[0]["reason"])

    def test_bad_quality_excluded(self):
        def reader(f, *a):
            r = good_reader()(f, *a)
            if f["i"] == 1:
                r["quality"] = {"ok": False, "problems": ["no finite values"]}
            return r
        _, ok, _, skipped = self.run_stream(reader)
        self.assertEqual(len(ok), 5); self.assertIn("no finite values", skipped[0]["reason"])

    def test_max_products_takes_latest(self):
        m, ok, arrays, skipped = self.run_stream(good_reader(), max_products=3)
        self.assertEqual(len(ok), 3)
        self.assertEqual(ok[-1]["product_id"], "TESTFIXTURE_5")

    def test_progress_called(self):
        seen = []
        self.run_stream(good_reader(), progress=lambda i, n, o: seen.append((i, n)))
        self.assertEqual(seen[0], (0, 6)); self.assertEqual(seen[-1], (5, 6))

    def test_opener_failure_reported(self):
        obs, row = make_world(4)
        def opener(url): raise ConnectionError("no route")
        m, ok, arrays, skipped = stream_stack(row, obs, 1, 1, opener=opener, reader=good_reader())
        self.assertEqual(ok, []); self.assertEqual(len(skipped), 4); self.assertIn("ConnectionError", skipped[0]["reason"])

    def test_pipeline_feeds_analysis(self):
        from modules.pulse_analysis import series_from_arrays
        m, ok, arrays, _ = self.run_stream(good_reader())
        s = series_from_arrays(arrays)
        self.assertEqual(s["n_obs"], 6); self.assertEqual(s["common_valid_pct"], 100.0)


class TestWindow(unittest.TestCase):
    x = np.arange(0, 2000, 20.0); y = np.arange(4000, 2000, -20.0)

    def test_centered_window(self):
        y0, y1, x0, x1 = window_indices(self.x, self.y, 1000.0, 3000.0, 10)
        self.assertEqual((x1 - x0, y1 - y0), (21, 21))
        self.assertEqual(self.x[x0 + 10], 1000.0); self.assertEqual(self.y[y0 + 10], 3000.0)

    def test_clipped_at_edge(self):
        y0, y1, x0, x1 = window_indices(self.x, self.y, 100.0, 3000.0, 50)
        self.assertEqual(x0, 0)

    def test_outside_footprint(self):
        with self.assertRaises(StreamError): window_indices(self.x, self.y, 9999.0, 3000.0, 10)
        with self.assertRaises(StreamError): window_indices(self.x, self.y, 1000.0, 99999.0, 10)

    def test_edge_window_too_small(self):
        with self.assertRaises(StreamError): window_indices(self.x, self.y, 0.0, 3000.0, 5)


class TestAuthAndCompare(unittest.TestCase):
    def test_login_without_credentials(self):
        for k in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
            os.environ.pop(k, None)
        with self.assertRaises(AuthError) as cm: earthaccess_login()
        self.assertIn("EARTHDATA_TOKEN", str(cm.exception))

    def test_unknown_maturity_both_is_warning_not_block(self):
        base = dict(product_type="GCOV", polarization="HH", maturity=UNKNOWN_TEXT, release="R1", direction="ASCENDING",
                    track=1, layer_path_leaf="HHHH", units="1", bbox=[0, 0, 1, 1],
                    first_acquisition="2026-01-01T00:00:00+00:00", last_acquisition="2026-03-01T00:00:00+00:00")
        r = compare_stacks(base, {**base, "track": 2, "bbox": [5, 5, 6, 6]})
        self.assertTrue(r["allowed"]); self.assertTrue(any("UNVERIFIED" in w for w in r["warnings"]))
        self.assertFalse(compare_stacks(base, {**base, "maturity": "VALIDATED"})["allowed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
