"""Walk-forward (rolling-origin) windows for the three-arm comparison: baseline / classical / quantum, the figure's
"Window 1 (train -> test) ... Window K", with the SAME seeds fed to every arm.

WHAT THIS ADDS (the codebase had two things, neither of them walk-forward)
  * temporal_validation.py   : 10 BLOCKS of the one fixed test period, from saved predictions (within-period stability).
  * run_feature_map_experiment: ONE train -> val -> test split.
  Here every window k RE-FITS each arm on all history before its own test window and scores it on the next, later window, so
  a result that only holds in one regime (one volatility episode, one test period) shows up as a window that disagrees.

DESIGN (all pure functions, numpy only, no torch)
  * Rows are the chronological concatenation [train; val; test] of data/processed (prepare_data keeps time order).
  * K test windows of `test_len` rows tile the END of the series, oldest first, so window K is the most recent data.
  * Window k trains on rows [train_start, fit_end) and picks its ridge on the last `val_frac` of the history before the
    test window, then leaves an EMBARGO of `gap` rows (>= 1) before test_start. Train pairs use targets up to fit_end-1, and
    test pairs start at test_start, so no target of any training pair is a test row (assert_no_leak checks it).
  * `expanding` (default) keeps all earlier history (capped by max_train_rows, chronological tail); `rolling` keeps a fixed length.
The decision rule of the pre-registration is NOT changed by this: walk-forward is a stability read-out, reported next to it.
"""
from __future__ import annotations

import math

import numpy as np

ARMS = ("linear", "classical_rf", "quantum")
ARM_ROLE = {"linear": "baseline", "classical_rf": "classical", "quantum": "quantum"}


def make_windows(n_rows: int, n_windows: int, test_len: int, *, min_train: int = 2000, val_frac: float = 0.2,
                 gap: int = 1, mode: str = "expanding", rolling_len: int | None = None,
                 max_train_rows: int | None = None) -> list[dict]:
    if n_windows < 1 or test_len < 30:
        raise ValueError("need n_windows >= 1 and test_len >= 30 rows")
    if gap < 1:
        raise ValueError("gap must be >= 1 (the target of the last training pair is the row after it)")
    if mode not in ("expanding", "rolling"):
        raise ValueError("mode must be 'expanding' or 'rolling'")
    first_test = n_rows - n_windows * test_len
    if first_test - gap < min_train:
        raise ValueError(f"not enough history: {n_windows} windows x {test_len} rows leave {first_test - gap} rows "
                         f"for training window 1 (< min_train {min_train}); lower --windows/--test-len")
    out = []
    for k in range(n_windows):
        test_start = first_test + k * test_len
        test_end = test_start + test_len
        hist_end = test_start - gap                       # rows < hist_end are usable history
        hist_start = 0
        if mode == "rolling":
            hist_start = max(0, hist_end - int(rolling_len or hist_end))
        if max_train_rows is not None:
            hist_start = max(hist_start, hist_end - int(max_train_rows))
        n_hist = hist_end - hist_start
        val_len = max(int(n_hist * val_frac), 30)
        fit_end = hist_end - val_len
        out.append({"window": k + 1, "train_start": hist_start, "train_end": fit_end, "val_start": fit_end,
                    "val_end": hist_end, "gap": gap, "test_start": test_start, "test_end": test_end,
                    "n_train": fit_end - hist_start, "n_val": val_len, "n_test": test_len})
    assert_no_leak(out)
    return out


def assert_no_leak(windows: list[dict]) -> None:
    """Chronology guard: train < val < embargo < test inside a window; test windows ascending and disjoint."""
    prev_end = -1
    for w in windows:
        if not (w["train_start"] < w["train_end"] <= w["val_start"] < w["val_end"] < w["test_start"] < w["test_end"]):
            raise AssertionError(f"window {w['window']}: train/val/test are not in chronological order")
        if w["test_start"] - w["val_end"] < w["gap"]:
            raise AssertionError(f"window {w['window']}: embargo shorter than gap")
        if w["test_start"] < prev_end:
            raise AssertionError(f"window {w['window']}: test window overlaps the previous one")
        prev_end = w["test_end"]


def concat_chronological(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def slice_window(X, y, w: dict):
    """(X_train, y_train, X_val, y_val, X_test, y_test) for one window."""
    return (X[w["train_start"]:w["train_end"]], y[w["train_start"]:w["train_end"]],
            X[w["val_start"]:w["val_end"]], y[w["val_start"]:w["val_end"]],
            X[w["test_start"]:w["test_end"]], y[w["test_start"]:w["test_end"]])


def summarise(per_run: list[dict], dm: list[dict], windows: list[dict], alpha: float = 0.05) -> dict:
    """Per window: median skill (vs persistence = the naive baseline) per arm across seeds, and in how many seeds the
    quantum arm beat each control (paired DM, one-sided: statistic > 0 with p < alpha). Overall: windows where it beat BOTH."""
    out = {"windows": []}
    for w in windows:
        k = w["window"]
        row = {"window": k, "test_start": w["test_start"], "test_end": w["test_end"], "n_train": w["n_train"], "skill_median": {},
               "n_seeds": len({r["seed"] for r in per_run if r["window"] == k})}
        for arm in ARMS:
            s = [r["skill"] for r in per_run if r["window"] == k and r["arm"] == arm and r.get("skill") is not None]
            row["skill_median"][arm] = float(np.median(s)) if s else None
        beat = {}
        for ctrl in ("linear", "classical_rf"):
            ds = [d for d in dm if d["window"] == k and d["vs"] == ctrl and "statistic" in d]
            beat[ctrl] = int(sum(1 for d in ds if d["statistic"] > 0 and _one_sided_p(d) < alpha))
            beat[ctrl + "_of"] = len(ds)
        row["quantum_beats"] = beat
        both = [d for d in dm if d["window"] == k and "statistic" in d]
        row["quantum_beats_both_in_all_seeds"] = bool(row["n_seeds"]) and all(
            beat[c] == beat[c + "_of"] == row["n_seeds"] for c in ("linear", "classical_rf")) and bool(both)
        out["windows"].append(row)
    wins = [w for w in out["windows"] if w["n_seeds"]]
    out["n_windows_run"] = len(wins)
    out["windows_quantum_beats_both"] = int(sum(w["quantum_beats_both_in_all_seeds"] for w in wins))
    out["windows_quantum_skill_positive"] = int(sum((w["skill_median"]["quantum"] or -1) > 0 for w in wins))
    out["windows_baseline_skill_positive"] = int(sum((w["skill_median"]["linear"] or -1) > 0 for w in wins))
    return out


def _one_sided_p(d: dict) -> float:
    p = d.get("p_value")
    if p is None:
        return 1.0
    return p / 2.0 if d["statistic"] > 0 else 1.0 - p / 2.0


def run_windows(X, y, windows, seeds, *, n_qubits=8, n_layers=3, entanglement="ring", input_scale=1.0,
                arms=ARMS, forecaster=None, dm_fn=None, log=print, attack_rows: int = 300, attack_eps: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Fit every arm on every window with the same seeds. `forecaster` / `dm_fn` are injectable (tests use cheap stand-ins).

    Every arm of one (window, seed) cell is scored with the SAME attack rows and latency rows: metric_seed = the cell's seed (13 Nov
    2026; before, this call used the default metric_seed=0 while scripts.run_naive_baseline used metric_seed=seed, so for seeds != 0
    Arm 1 attacked different rows than Arms 2 and 3)."""
    if forecaster is None:
        from src.baselines.feature_map_forecaster import FeatureMapForecaster as forecaster  # noqa: N813
    if dm_fn is None:
        from src.evaluation.advantage_diagnostics import diebold_mariano as dm_fn
    per_run, dm = [], []
    for w in windows:
        Xtr, ytr, Xva, yva, Xte, yte = slice_window(X, y, w)
        for seed in seeds:
            errs = {}
            for arm in arms:
                m = forecaster(arm, n_qubits, n_layers, entanglement, input_scale, seed)
                r = m.fit(Xtr, ytr, Xva, yva).evaluate(Xte, yte, attack_rows=attack_rows, attack_eps=attack_eps, metric_seed=seed)
                errs[arm] = r.pop("errors")
                r.update({"seed": seed, "window": w["window"]})
                per_run.append(r)
                log(f"window {w['window']} seed {seed} {arm:13s} skill={r['skill']:+.5f} rmse={r['rmse']:.6g} "
                    f"persistence={r['persistence_rmse']:.6g}")
            for other in ("linear", "classical_rf"):
                if other in errs and "quantum" in errs:
                    d = dict(dm_fn(errs[other], errs["quantum"]))
                    d.update({"seed": seed, "vs": other, "window": w["window"]})
                    dm.append(d)
    return per_run, dm
