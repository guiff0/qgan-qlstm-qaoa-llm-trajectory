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
