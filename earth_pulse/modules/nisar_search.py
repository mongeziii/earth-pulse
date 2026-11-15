"""Archive discovery: windowed search with truncation detection, bisection, progress and a call budget."""
from datetime import timedelta

from .config import SEARCH_CAP, MIN_SPLIT_SPAN_HOURS

DEFAULT_MAX_CALLS = 200


def search_interval(search_fn, start, end, cap=SEARCH_CAP, min_span_hours=MIN_SPLIT_SPAN_HOURS,
                    budget=None, on_window=None):
    """search_fn(start, end, cap) -> (records, search_complete).
    A window is 'truncated' if it returned >= cap records or the API said the search was incomplete;
    truncated windows are bisected. A window that cannot be resolved (minimum span reached, or the
    call budget exhausted) is reported as unresolved and the search is then NOT complete.
    budget: shared dict {'calls': int, 'max': int}. on_window(info) is called after every request."""
    budget = budget if budget is not None else {"calls": 0, "max": DEFAULT_MAX_CALLS}
    windows, unresolved, records = [], [], []
    min_span = timedelta(hours=min_span_hours)

    def rec(s, e):
        if budget["calls"] >= budget["max"]:
            w = {"start": s.isoformat(), "end": e.isoformat(), "returned": 0, "truncated": True,
                 "split": False, "unresolved": True, "reason": "call budget exhausted"}
            windows.append(w); unresolved.append({k: w[k] for k in ("start", "end", "returned", "reason")})
            return
        budget["calls"] += 1
        recs, complete = search_fn(s, e, cap)
        truncated = len(recs) >= cap or complete is False
        w = {"start": s.isoformat(), "end": e.isoformat(), "returned": len(recs),
             "truncated": truncated, "split": False, "unresolved": False}
        windows.append(w)
        if on_window:
            on_window({**w, "call": budget["calls"], "max_calls": budget["max"]})
        if not truncated:
            records.extend(recs)
        elif (e - s) <= min_span:
            records.extend(recs)
            w["unresolved"] = True
            unresolved.append({"start": w["start"], "end": w["end"], "returned": len(recs)})
        else:
            w["split"] = True
            mid = s + (e - s) / 2
            rec(s, mid)
            rec(mid, e)

    rec(start, end)
    return records, {"windows": windows, "unresolved_truncated_windows": unresolved,
                     "complete": not unresolved, "calls_used": budget["calls"]}


def search_archive(search_fn, start, end, chunk_days=30, cap=SEARCH_CAP, max_calls=DEFAULT_MAX_CALLS,
                   on_window=None):
    """Search [start, end] in chunk_days intervals under ONE shared call budget."""
    budget = {"calls": 0, "max": max_calls}
    all_records, windows, unresolved = [], [], []
    s = start
    while s < end:
        e = min(s + timedelta(days=chunk_days), end)
        recs, rep = search_interval(search_fn, s, e, cap, budget=budget, on_window=on_window)
        all_records.extend(recs)
        windows.extend(rep["windows"])
        unresolved.extend(rep["unresolved_truncated_windows"])
        s = e
    return all_records, {"n_windows": len(windows), "windows": windows,
                         "unresolved_truncated_windows": unresolved, "complete": not unresolved,
                         "raw_record_count": len(all_records), "calls_used": budget["calls"],
                         "call_budget": max_calls,
                         "budget_exhausted": budget["calls"] >= max_calls and bool(unresolved)}
