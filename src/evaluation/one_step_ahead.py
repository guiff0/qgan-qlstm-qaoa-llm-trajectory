"""
Fix for the same-row target leakage flagged in the earlier audit and
detected (but never fixed) by scripts/prepare_data.py's leakage_check().

X_train.npy/y_train.npy etc. keep y[k] == close[k] (same-row) on disk,
because Classical LSTM's WindowedSequenceDataset genuinely needs that
exact convention (see prepare_data.py's comment on this). But Classical
GAN-LLM, QGAN-LLM, and QLSTM Forecaster all train on (X[k], y[k]) pairs
directly, with no windowing step in between -- for them, same-row
pairing means the target is sitting right there among X[k]'s own
price-derived indicator features (RSI, MACD, SMA, etc., all computed
FROM close[k]). That is not a one-step-ahead forecast.

shift_for_one_step_ahead(X, y) turns the SAME loaded arrays into a
genuine one-step-ahead problem locally, without touching the files on
disk or Classical LSTM's own (already-correct) use of them: pair X[k]
with y[k+1] instead, dropping the final now-unlabeled row. Call this
once, right after loading X_train/y_train (and X_val/y_val, X_test/
y_test), before doing anything else with them.
"""
from __future__ import annotations

import numpy as np


def shift_for_one_step_ahead(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X[:-1], y[1:]) -- row k of the output pairs k's features
    with the target that was originally at row k+1. Raises if X and y
    aren't the same length (same contract as WindowedSequenceDataset's
    own check) or if there are fewer than 2 rows (nothing to shift)."""
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if len(X) < 2:
        raise ValueError(f"need at least 2 rows to form a one-step-ahead pair, got {len(X)}")
    return X[:-1], y[1:]


# ---------------------------------------------------------------------------
# Forecast target frame: predict the next-close CHANGE, not the price level.
#
# Why: the models see 32 PCA components of ~54 features, which cannot reproduce the current close
# exactly, and the train (2012-2021) and test (2024-2026) price levels differ. A linear head that
# regresses the LEVEL therefore carries reconstruction + level-shift error and landed at ~4.5x the
# persistence RMSE (next close = current close). In the "delta" frame the target is
# y[k+1] - y[k]; a head that learns nothing predicts 0 and scores EXACTLY persistence, so any
# real signal shows up as skill above it. RMSE in the delta frame equals RMSE of the price
# forecast (last close + delta) against the next close, so it is directly comparable with the
# persistence RMSE and with older level-frame rows.
# ---------------------------------------------------------------------------
from typing import NamedTuple

FRAMES = ("delta", "level")


class ShiftedTarget(NamedTuple):
    X: np.ndarray           # X[:-1]
    target: np.ndarray      # what the model is trained against / scored on in this frame
    y_next: np.ndarray      # next close (price level), y[1:]
    last: np.ndarray        # last observed close, y[:-1]
    reference: np.ndarray   # "last price" for the directional-flip attack metric in this frame
    frame: str


def resolve_frame(config: dict) -> str:
    frame = str((config or {}).get("forecast_target", "delta")).lower()
    if frame not in FRAMES:
        raise ValueError(f"forecast_target must be one of {FRAMES}, got {frame!r}")
    return frame


def make_target(X: np.ndarray, y: np.ndarray, frame: str = "delta") -> ShiftedTarget:
    """One-step-ahead pairing in the requested frame. Differences are taken in float64 (prices ~1.1
    with changes ~1e-4 would lose precision in float32).
      delta: target = y[k+1] - y[k]; attack reference = 0  (flip = sign change of the forecast change)
      level: target = y[k+1];        attack reference = y[k] (the original behaviour)"""
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    X1, _ = shift_for_one_step_ahead(np.asarray(X), np.asarray(y))
    y64 = np.asarray(y, dtype=np.float64).reshape(len(y), -1)[:, 0]
    y_next, last = y64[1:], y64[:-1]
    if frame == "delta":
        return ShiftedTarget(X1, y_next - last, y_next, last, np.zeros_like(last), frame)
    return ShiftedTarget(X1, y_next, y_next, last, last, frame)
