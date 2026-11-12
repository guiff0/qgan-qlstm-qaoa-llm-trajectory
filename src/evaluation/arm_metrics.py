"""The five metrics of the methodology figure, computed IDENTICALLY for every arm (inputs-only baseline, classical random
features, quantum circuit features):  persistence skill (s), RMSE, adversarial ASR, forecast fidelity, latency.

Pure numpy. A model only has to expose `predict_residual(X) -> (n,)` (the one-step change over persistence), so the same
function scores the ridge baseline, the classical map and the circuit with the SAME attack, rows and clock.

  asr        FGSM-style directional-flip rate. Every input column is shifted by eps x (training SD of that column) in the
             sign of the finite-difference gradient of the forecast, so the attack goes through whatever sits on the
             inference path (for the quantum arm: through the circuit). Success = the sign of the forecast change flips.
             Central differences need 2 x d forward passes on `n_rows` rows (a model-agnostic stand-in for autograd).
  fidelity   1 - W1(pred residuals, true residuals) / SD(true residuals), clipped to [0, 1]; higher is better. It is a
             FORECAST-distribution fidelity defined for every arm. It is NOT the generator FID/MMD of the GAN study; the
             two must not be pooled (the compliance check records which one a row carries).
  latency    wall-clock ms for ONE row (batch size 1) through the full inference path, after warm-up.
"""
from __future__ import annotations

import time

import numpy as np

METRIC_KEYS = ("skill", "rmse", "asr", "fidelity", "latency_ms")


def _grad_sign(predict, X: np.ndarray, step: np.ndarray) -> np.ndarray:
    G = np.zeros_like(X)
    for j in range(X.shape[1]):
        Xp, Xm = X.copy(), X.copy()
        Xp[:, j] += step[j]
        Xm[:, j] -= step[j]
        G[:, j] = predict(Xp) - predict(Xm)
    return np.sign(G)


def adversarial_asr(predict, X, col_sd, eps: float = 0.1, n_rows: int = 300, seed: int = 0) -> dict:
    """Directional-flip ASR in % on a seeded subsample, so every arm attacks the SAME rows."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=min(n_rows, len(X)), replace=False))
    Xs = X[idx]
    sd = np.asarray(col_sd, dtype=np.float64)
    base = predict(Xs)
    s = _grad_sign(predict, Xs, 0.5 * eps * sd)
    adv = predict(Xs + eps * sd * s)
    flip = np.sign(base) != np.sign(adv)
    return {"asr": float(100.0 * flip.mean()), "asr_n": int(len(idx)), "asr_eps": float(eps),
            "asr_mean_pred_shift": float(np.mean(np.abs(adv - base)))}


def forecast_fidelity(pred_resid, true_resid) -> float:
    from scipy.stats import wasserstein_distance
    t = np.asarray(true_resid, dtype=np.float64)
    sd = float(t.std())
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    w1 = wasserstein_distance(np.asarray(pred_resid, dtype=np.float64), t)
    return float(min(1.0, max(0.0, 1.0 - w1 / sd)))


def latency_ms(predict, X, repeats: int = 50, warmup: int = 5, seed: int = 0) -> dict:
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(X), size=repeats + warmup)
    for r in rows[:warmup]:
        predict(X[r:r + 1])
    t = []
    for r in rows[warmup:]:
        t0 = time.perf_counter()
        predict(X[r:r + 1])
        t.append((time.perf_counter() - t0) * 1e3)
    return {"latency_ms": float(np.mean(t)), "latency_sd_ms": float(np.std(t, ddof=1)) if len(t) > 1 else float("nan")}
