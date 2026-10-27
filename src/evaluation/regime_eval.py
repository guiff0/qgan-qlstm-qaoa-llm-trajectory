"""
Per-regime evaluation, so an ANCOVA with VIX as the covariate has something to
vary over.

Why this exists: every model is evaluated on one shared test split, so a
single per-run VIX value is the same number for every run and the ANCOVA
covariate is constant (the plain-t-test fallback in run_hypothesis_tests.py).
Splitting each run's test error by VIX regime gives each run one row per
regime, each with its own mean VIX -- the covariate then varies.

Regimes follow Ch.3: low (VIX < 15), normal (15 <= VIX < 25), high (VIX >= 25).
VIX values come from data/processed/vix_<split>.npy, written by prepare_data.py
(the publication-lagged series already merged onto the minute bars).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

LOW, HIGH = 15.0, 25.0


def regime_labels(vix: np.ndarray) -> np.ndarray:
    v = np.asarray(vix, dtype=float)
    return np.where(v < LOW, "low", np.where(v < HIGH, "normal", "high"))


def rmse_by_regime(y_true: np.ndarray, y_pred: np.ndarray, vix: np.ndarray, min_rows: int = 100) -> pd.DataFrame:
    """One row per regime: n, mean VIX, RMSE, MAE. Regimes with fewer than
    min_rows observations are reported with NaN errors rather than a noisy number."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float); vix = np.asarray(vix, float)
    if not (len(y_true) == len(y_pred) == len(vix)):
        raise ValueError(f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, vix={len(vix)}")
    lab = regime_labels(vix)
    rows = []
    for r in ("low", "normal", "high"):
        m = lab == r
        n = int(m.sum())
        err = y_true[m] - y_pred[m]
        ok = n >= min_rows
        rows.append({"regime": r, "n": n, "mean_vix": float(vix[m].mean()) if n else float("nan"),
                     "rmse": float(np.sqrt(np.mean(err ** 2))) if ok else float("nan"),
                     "mae": float(np.mean(np.abs(err))) if ok else float("nan")})
    return pd.DataFrame(rows)


def stack_for_ancova(per_run: dict[str, pd.DataFrame], metric: str = "rmse") -> pd.DataFrame:
    """per_run: {condition_label: rmse_by_regime() frame} -> long frame with
    columns dv, group, covariate ready for statistical_tests.ancova()."""
    out = []
    for label, df in per_run.items():
        d = df.dropna(subset=[metric, "mean_vix"])
        out.append(pd.DataFrame({"dv": d[metric].to_numpy(), "group": label, "covariate": d["mean_vix"].to_numpy(), "regime": d["regime"].to_numpy()}))
    return pd.concat(out, ignore_index=True)
