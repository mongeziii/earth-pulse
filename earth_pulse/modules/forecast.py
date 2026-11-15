"""EARTH PULSE FORECAST: estimate the NEXT measured mean-backscatter value of a stack.

This forecasts a measured radar signal, not an event. It does not predict floods, earthquakes
or any other disaster, and implies no cause.

DESIGN
* Series: one value per REAL acquisition (mean dB over common valid pixels, from pulse_analysis).
  No interpolation, no synthetic dates, no imputed observations.
* Row t (t >= 1) predicts the change delta_t = v[t] - v[t-1] from information available at t-1 ONLY:
  last_value, last_change, rate (db/day), rolling_mean3, rolling_std3, interval_days (known
  acquisition gap), last_valid_pct. Forecast level = v[t-1] + predicted delta.
* Models: persistence baseline (delta = 0), LinearRegression, RandomForestRegressor(seed fixed).
* Chronology: held-out TEST = the last n_test rows. Model SELECTION uses walk-forward validation
  inside the TRAINING rows only; the test rows are used once, afterwards, for evaluation.
  Walk-forward: to predict row i, fit only on rows < i. A model replaces persistence only if it has
  lower validation RMSE than persistence; otherwise persistence is selected.
* Uncertainty: empirical interval of absolute walk-forward residuals, only if >= MIN_RESIDUALS_FOR_INTERVAL
  residuals exist. Approximate; coverage is NOT guaranteed.
* Insufficient history => status 'unavailable' and no model is trained.
"""
import hashlib
import json
from datetime import timedelta
from statistics import median

import numpy as np

from .product_metadata import parse_time

MIN_OBS = 10                    # >= 9 rows: 3 test + 6 train (3 to seed, 3 to validate)
MIN_TRAIN_ROWS = 6
MIN_SEED_ROWS = 3               # rows needed before the first validation prediction
MIN_RESIDUALS_FOR_INTERVAL = 10
INTERVAL_QUANTILE = 0.9
SEED = 42
FEATURES = ("last_value", "last_change", "rate_db_per_day", "rolling_mean3", "rolling_std3",
            "interval_days", "last_valid_pct")
UNAVAILABLE_MSG = "Forecast unavailable: insufficient compatible historical observations."


def build_dataset(dates, mean_db, valid_pct):
    """Return (X, y_delta, last, idx). Row r (t=r+1) uses observations 0..t-1 for features and
    observation t only for its label. dates: ISO strings; all inputs time-ordered."""
    v = np.asarray(mean_db, float)
    t = [parse_time(d) for d in dates]
    if any(x is None for x in t) or len(v) != len(t) or len(valid_pct) != len(v):
        raise ValueError("dates/values/valid_pct must be complete and equal length")
    if any(b <= a for a, b in zip(t, t[1:])):
        raise ValueError("acquisitions must be strictly increasing in time")
    X, y, last, idx = [], [], [], []
    for k in range(1, len(v)):                       # k = target observation index
        j = k - 1                                    # latest observation usable as a feature
        gap_prev = (t[j] - t[j - 1]).total_seconds() / 86400 if j >= 1 else np.nan
        chg = v[j] - v[j - 1] if j >= 1 else 0.0
        win = v[max(0, j - 2): j + 1]
        gap_target = (t[k] - t[j]).total_seconds() / 86400
        X.append([v[j], chg, chg / gap_prev if j >= 1 else 0.0, float(win.mean()),
                  float(win.std()), gap_target, float(valid_pct[j])])
        y.append(v[k] - v[j]); last.append(v[j]); idx.append(k)
    return np.array(X), np.array(y), np.array(last), idx


def _model(name):
    if name == "persistence":
        return None
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    return LinearRegression() if name == "linear_regression" else \
        RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1)


def _predict_delta(name, X_tr, y_tr, X_new):
    if name == "persistence":
        return np.zeros(len(X_new))
    m = _model(name)
    m.fit(X_tr, y_tr)
    return m.predict(X_new)


def walk_forward(name, X, y, rows):
    """Predict each row in `rows` fitting only on strictly earlier rows. Returns (pred_delta, used_train_sizes)."""
    preds, sizes = [], []
    for i in rows:
        assert i >= 1 or name == "persistence"
        preds.append(float(_predict_delta(name, X[:i], y[:i], X[i:i + 1])[0]))
        sizes.append(i)
    return np.array(preds), sizes


def _metrics(actual_delta, pred_delta):
    e = np.asarray(actual_delta) - np.asarray(pred_delta)   # level error == delta error
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "n": int(len(e))}


def run_forecast(dates, mean_db, valid_pct, product_ids):
    n = len(mean_db)
    if n < MIN_OBS:
        return {"status": "unavailable", "message": UNAVAILABLE_MSG,
                "reason": f"{n} observations available; {MIN_OBS} required"}
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return {"status": "unavailable", "message": UNAVAILABLE_MSG, "reason": "scikit-learn not installed"}
    X, y, last, idx = build_dataset(dates, mean_db, valid_pct)
    rows = len(y)
    n_test = max(2, int(round(0.3 * rows)))
    n_train = rows - n_test
    if n_train < MIN_TRAIN_ROWS:
        return {"status": "unavailable", "message": UNAVAILABLE_MSG,
                "reason": f"only {n_train} training rows after holding out {n_test}; {MIN_TRAIN_ROWS} required"}
    names = ("persistence", "linear_regression", "random_forest")
    val_rows = list(range(MIN_SEED_ROWS, n_train))
    test_rows = list(range(n_train, rows))
    val, test, resid = {}, {}, {}
    for nm in names:
        pv, _ = walk_forward(nm, X, y, val_rows)
        pt, _ = walk_forward(nm, X, y, test_rows)
        val[nm] = _metrics(y[val_rows], pv)
        test[nm] = _metrics(y[test_rows], pt)
        resid[nm] = np.concatenate([y[val_rows] - pv, y[test_rows] - pt])
    selected = "persistence"
    for nm in ("linear_regression", "random_forest"):    # selection uses VALIDATION only
        if val[nm]["rmse"] < val[selected]["rmse"]:
            selected = nm
    gaps = [(parse_time(b) - parse_time(a)).total_seconds() / 86400 for a, b in zip(dates, dates[1:])]
    horizon = float(median(gaps))
    t_last = parse_time(dates[-1])
    j = n - 1
    chg = mean_db[j] - mean_db[j - 1]
    win = np.asarray(mean_db[-3:], float)
    x_next = np.array([[mean_db[j], chg, chg / gaps[-1], float(win.mean()), float(win.std()), horizon,
                        float(valid_pct[j])]])
    delta = float(_predict_delta(selected, X, y, x_next)[0])
    point = float(mean_db[j] + delta)
    r = resid[selected]
    interval = None
    if len(r) >= MIN_RESIDUALS_FOR_INTERVAL:
        q = float(np.quantile(np.abs(r), INTERVAL_QUANTILE))
        interval = {"lower": point - q, "upper": point + q, "level": INTERVAL_QUANTILE, "n_residuals": int(len(r)),
                    "method": "empirical quantile of absolute walk-forward residuals; approximate, "
                              "coverage not guaranteed"}
    fp = hashlib.sha256(json.dumps([list(product_ids), [round(float(x), 9) for x in mean_db], SEED,
                                    MIN_OBS], sort_keys=True).encode()).hexdigest()[:16]
    t_idx = lambda r_: [dates[idx[i]][:10] for i in r_]
    return {
        "status": "ok", "label": "MODEL ESTIMATE - not a NASA observation",
        "n_observations": n, "n_rows": rows, "n_train_rows": n_train, "n_test_rows": n_test,
        "training_period": [dates[idx[0]][:10], dates[idx[n_train - 1]][:10]],
        "test_period": [dates[idx[n_train]][:10], dates[idx[-1]][:10]],
        "test_dates": t_idx(test_rows),
        "validation_metrics": val, "test_metrics": test,
        "selected_model": selected, "selection_basis": "lowest walk-forward validation RMSE inside the training rows; "
                                                       "persistence unless a model beats it",
        "beats_persistence_on_test": {nm: bool(test[nm]["rmse"] < test["persistence"]["rmse"] and
                                               test[nm]["mae"] < test["persistence"]["mae"])
                                      for nm in names if nm != "persistence"},
        "forecast": {"value_db": point, "predicted_change_db": delta,
                     "forecast_date": (t_last + timedelta(days=horizon)).date().isoformat(),
                     "horizon_days": horizon,
                     "horizon_assumption": "next acquisition assumed one median historical interval after the "
                                           "last observation; the real acquisition date is not known",
                     "interval": interval,
                     "interval_note": None if interval else
                     f"No uncertainty interval: {len(r)} walk-forward residuals, "
                     f"{MIN_RESIDUALS_FOR_INTERVAL} required for the empirical method."},
        "last_observation": {"date": dates[-1][:10], "value_db": float(mean_db[-1]), "product_id": product_ids[-1]},
        "product_ids": list(product_ids), "features": list(FEATURES), "seed": SEED, "data_fingerprint": fp,
        "limitations": [
            "Forecasts the mean backscatter of a fixed sampled area, not events; no causal claim.",
            f"Very small training set ({rows} rows); estimates have high variance.",
            "Backscatter varies with moisture, vegetation, surface and geometry changes the model cannot see.",
            "Selection used validation rows; held-out rows were used only for evaluation, but with this few rows "
            "test metrics are themselves noisy.",
            "Empirical intervals (if any) are approximate and may under-cover."]}


def forecast_explanation(res):
    if res["status"] != "ok":
        return f"{res['message']} ({res['reason']})."
    f, sel = res["forecast"], res["selected_model"]
    tm = res["test_metrics"]
    beat = res["beats_persistence_on_test"].get(sel)
    perf = (f"On {res['n_test_rows']} held-out later observations the {sel.replace('_', ' ')} model had MAE "
            f"{tm[sel]['mae']:.2f} dB and RMSE {tm[sel]['rmse']:.2f} dB, versus the persistence baseline's "
            f"MAE {tm['persistence']['mae']:.2f} dB and RMSE {tm['persistence']['rmse']:.2f} dB. ")
    if sel == "persistence":
        pat = ("No trained model beat the persistence baseline on validation, so the forecast simply carries the "
               "last measured value forward; no learned pattern is claimed. ")
    else:
        pat = (f"The {sel.replace('_', ' ')} model, trained on earlier observations, outperformed persistence on "
               f"validation and is used to estimate the next change. "
               + ("It also beat persistence on the held-out test rows. " if beat else
                  "It did NOT beat persistence on the held-out test rows, so treat it with extra caution. "))
    iv = f["interval"]
    ivt = (f"An approximate {iv['level']:.0%} empirical interval is {iv['lower']:.2f} to {iv['upper']:.2f} dB. "
           if iv else f["interval_note"] + " ")
    return (f"This forecast uses {res['n_observations']} real NISAR observations ({res['training_period'][0]} to "
            f"{res['test_period'][1]}). It estimates the mean backscatter at the next acquisition, assumed "
            f"{f['horizon_days']:.0f} days after the last one ({f['forecast_date']}), as {f['value_db']:.2f} dB "
            f"(last measured {res['last_observation']['value_db']:.2f} dB). " + pat + perf + ivt +
            "This is a model estimate, not a NASA observation. It could be wrong if surface conditions change "
            "(rain, vegetation, land use, acquisition geometry) in ways absent from past data. No cause is implied.")
