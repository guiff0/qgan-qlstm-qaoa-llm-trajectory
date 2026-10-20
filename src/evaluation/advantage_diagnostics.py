"""
Diagnostics for "why is no quantum advantage visible?".

Pure numpy / pandas -- importable without torch or PennyLane, so the checks
can run on the artifacts a finished pipeline leaves behind (results CSV,
*.synth.npy) and inside unit tests.

Each helper answers one concrete question that, in the Oct 12-17, 2026 runs,
turned out to explain part of the null result (see CHANGES_OCT19.md and
Chapter 4, "Why No Quantum Advantage Is Observed"):

  1. Is the comparison degenerate?   -> find_degenerate_groups
  2. Can the circuit reach the output? -> generator_rank_bound, numerical_rank,
                                          effective_rank
  3. Is the training budget real?    -> training_budget_fraction
  4. Is capacity matched?            -> lstm_generator_params, qgan_generator_params
  5. Is there signal to find?        -> skill_vs_persistence
  6. Is a difference significant?    -> diebold_mariano
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np


# --------------------------------------------------------------------------
# 1. Degenerate comparisons
# --------------------------------------------------------------------------
def find_degenerate_groups(values_by_group: Dict[str, Iterable[float]],
                           rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """Detect the case where every group produced the same number.

    Returns {"degenerate": bool, "reason": str, "n_unique": int}. A
    zero-variance comparison cannot support a test statistic; reporting it
    explicitly is better than a NaN t-statistic."""
    flat = []
    for vals in values_by_group.values():
        flat.extend(float(v) for v in np.asarray(list(vals), dtype=float).ravel()
                    if np.isfinite(v))
    if len(flat) < 2:
        return {"degenerate": True, "reason": "fewer than 2 finite observations", "n_unique": len(flat)}
    arr = np.asarray(flat)
    spread = float(arr.max() - arr.min())
    scale = max(abs(float(arr.mean())), atol)
    if spread <= atol + rtol * scale:
        return {"degenerate": True,
                "reason": f"all {len(arr)} observations are identical "
                          f"(value={arr[0]:.6g}); the quantity does not depend on the model "
                          f"under test, so no test statistic exists",
                "n_unique": 1}
    return {"degenerate": False, "reason": "", "n_unique": int(len(np.unique(np.round(arr, 12))))}


# --------------------------------------------------------------------------
# 2. Can the circuit reach the output? (rank of what the generator can emit)
# --------------------------------------------------------------------------
def generator_rank_bound(n_qubits: int, n_features: int, readout: str = "z") -> int:
    """Upper bound on the affine rank of the QLSTM generator's output.

    The generator is  y = W * <Z>(z) + b  with <Z> an n_qubit vector of
    Pauli-Z expectation values, so every output lies in an affine subspace of
    dimension <= n_qubits (rank of the centred data <= n_qubits). With a
    'z+zz' readout (not implemented in the repo) the bound would be larger."""
    readout = readout.lower()
    if readout == "z":
        reach = n_qubits
    elif readout == "z+zz":
        reach = n_qubits + n_qubits * (n_qubits - 1) // 2
    else:
        raise ValueError(f"unknown readout {readout!r}")
    return int(min(reach, n_features))


def numerical_rank(X: np.ndarray, rel_tol: float = 1e-6) -> int:
    """Rank of the centred data matrix (singular values above rel_tol * max)."""
    X = np.asarray(X, dtype=np.float64)
    s = np.linalg.svd(X - X.mean(axis=0, keepdims=True), compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    return int((s > rel_tol * s[0]).sum())


def effective_rank(X: np.ndarray) -> float:
    """exp(entropy of the normalised eigenvalue spectrum) (Roy & Vetterli 2007)."""
    X = np.asarray(X, dtype=np.float64)
    s = np.linalg.svd(X - X.mean(axis=0, keepdims=True), compute_uv=False)
    ev = s ** 2
    tot = ev.sum()
    if tot <= 0:
        return 0.0
    p = ev / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# --------------------------------------------------------------------------
# 3. Training budget
# --------------------------------------------------------------------------
def training_budget_fraction(max_batches_per_epoch: Optional[int], batch_size: int,
                             epochs: int, n_train_rows: int) -> dict:
    """Fraction of the training set a capped run can touch.

    rows_per_epoch = max_batches * batch_size; a run sees at most
    rows_per_epoch * epochs row-visits but only rows_per_epoch * epochs
    DISTINCT rows if each epoch draws fresh ones (upper bound)."""
    if not max_batches_per_epoch:
        rows_per_epoch = n_train_rows
    else:
        rows_per_epoch = min(n_train_rows, int(max_batches_per_epoch) * int(batch_size))
    visits = rows_per_epoch * int(epochs)
    return {"rows_per_epoch": int(rows_per_epoch),
            "max_row_visits": int(visits),
            "max_distinct_rows": int(min(n_train_rows, visits)),
            "fraction_of_train_set": float(min(1.0, visits / max(n_train_rows, 1)))}


# --------------------------------------------------------------------------
# 4. Capacity matching
# --------------------------------------------------------------------------
def lstm_generator_params(latent_dim: int = 32, hidden_dim: int = 128, output_dim: int = 32) -> int:
    """Trainable parameters of ClassicalGANLLM's LSTMGenerator (1-layer LSTM + Linear)."""
    lstm = 4 * hidden_dim * (latent_dim + hidden_dim) + 2 * 4 * hidden_dim  # W_ih, W_hh, b_ih, b_hh
    fc = hidden_dim * output_dim + output_dim
    return int(lstm + fc)


def qgan_generator_params(n_qubits: int = 20, n_layers: int = 4, n_features: int = 32) -> int:
    """Trainable parameters of QLSTMGenerator: 3 rotation angles per qubit per
    layer + the classical Linear(n_qubits -> n_features) read-out."""
    return int(n_layers * n_qubits * 3 + n_qubits * n_features + n_features)


# --------------------------------------------------------------------------
# 5. Is there signal beyond persistence?
# --------------------------------------------------------------------------
def skill_vs_persistence(y_true_next: np.ndarray, y_pred_next: np.ndarray,
                         y_current: np.ndarray) -> dict:
    """RMSE, persistence RMSE and the MSE skill score 1 - MSE/MSE_persistence.

    Skill > 0 means the model beats 'next close = current close'."""
    y_true_next = np.asarray(y_true_next, dtype=np.float64).ravel()
    y_pred_next = np.asarray(y_pred_next, dtype=np.float64).ravel()
    y_current = np.asarray(y_current, dtype=np.float64).ravel()
    if not (len(y_true_next) == len(y_pred_next) == len(y_current)):
        raise ValueError("length mismatch")
    mse = float(np.mean((y_true_next - y_pred_next) ** 2))
    mse_p = float(np.mean((y_true_next - y_current) ** 2))
    return {"rmse": math.sqrt(mse), "persistence_rmse": math.sqrt(mse_p),
            "skill": (1.0 - mse / mse_p) if mse_p > 0 else float("nan")}


# --------------------------------------------------------------------------
# 6. Paired test on forecast errors
# --------------------------------------------------------------------------
def diebold_mariano(e1: np.ndarray, e2: np.ndarray, lag: Optional[int] = None) -> dict:
    """Diebold-Mariano test of equal squared-error loss, one-step horizon,
    Newey-West (Bartlett) long-run variance.

    Positive statistic => model 1 has LARGER loss than model 2 (model 2 better).
    Both error series must come from the same test rows."""
    e1 = np.asarray(e1, dtype=np.float64).ravel()
    e2 = np.asarray(e2, dtype=np.float64).ravel()
    if len(e1) != len(e2):
        raise ValueError("error series must be paired (same length)")
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 10:
        return {"error": f"need >= 10 paired errors, have {n}"}
    dbar = float(d.mean())
    if np.allclose(d, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "mean_loss_diff": 0.0, "n": n,
                "note": "identical forecasts"}
    if lag is None:
        lag = int(math.floor(n ** (1.0 / 3.0)))
    dc = d - dbar
    lrv = float(np.dot(dc, dc) / n)
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * float(np.dot(dc[k:], dc[:-k]) / n)
    lrv = max(lrv, 1e-30)
    stat = dbar / math.sqrt(lrv / n)
    p = math.erfc(abs(stat) / math.sqrt(2.0))  # two-sided normal
    return {"statistic": float(stat), "p_value": float(p), "mean_loss_diff": dbar, "n": n, "lag": lag}
