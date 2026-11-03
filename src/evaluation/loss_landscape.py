"""
Loss-landscape diagnostics for a trained generator (the code behind the paper's loss-landscape figure).

Everything here is written against plain callables on a flat NumPy parameter vector, so it is independent
of PyTorch/PennyLane and unit-testable on analytic losses (tests/test_loss_landscape.py):

    loss_fn(theta)  -> float            scalar loss at parameters theta
    grad_fn(theta)  -> np.ndarray       gradient of that loss (autograd / adjoint differentiation)

What it computes
  * slice_2d        loss on a 2-D plane theta + a*d1 + b*d2 (Li et al. 2018 style, filter-normalised directions).
  * hessian_top_eig largest-magnitude Hessian eigenvalue ("sharpness") by power iteration, with Hessian-vector
                    products from central differences of grad_fn.  Also a Hutchinson estimate of trace(H).
  * roughness       mean |second difference| along a 1-D line: how jagged the loss is along a random direction.

What it does NOT compute: a Hessian *condition number* (the smallest eigenvalue of a barely-trained circuit is
dominated by numerical noise, so the ratio would be meaningless), or a local-minima density.  The result is a
description of ONE trained generator on ONE fixed batch; it is not evidence that entanglement helps.
The loss must be deterministic in theta (fix any sampling noise inside loss_fn), or finite differences are garbage.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

LossFn = Callable[[np.ndarray], float]
GradFn = Callable[[np.ndarray], np.ndarray]


def random_direction(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian direction rescaled to ||theta|| (a single-block filter normalisation)."""
    d = rng.standard_normal(theta.shape)
    n_t = float(np.linalg.norm(theta))
    return d * (n_t / max(float(np.linalg.norm(d)), 1e-12)) if n_t > 0 else d / max(float(np.linalg.norm(d)), 1e-12)


def slice_2d(loss_fn: LossFn, theta: np.ndarray, d1: np.ndarray, d2: np.ndarray,
             span: float = 0.5, n: int = 21) -> Dict:
    """Loss on the grid theta + a*d1 + b*d2, a,b in [-span, span] (fractions of ||theta||). Returns a JSON-able dict."""
    axis = np.linspace(-span, span, n)
    Z = np.empty((n, n))
    for i, a in enumerate(axis):
        for j, b in enumerate(axis):
            Z[i, j] = loss_fn(theta + a * d1 + b * d2)
    return {"axis": axis.tolist(), "loss": Z.tolist(), "center_loss": float(loss_fn(theta)),
            "grid_min": float(Z.min()), "grid_max": float(Z.max()), "span": span, "n": n}


def hvp(grad_fn: GradFn, theta: np.ndarray, v: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central-difference Hessian-vector product H v ~ (g(theta+eps*u) - g(theta-eps*u)) / (2 eps) * |v|, u = v/|v|."""
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        return np.zeros_like(v)
    u = v / nv
    return (grad_fn(theta + eps * u) - grad_fn(theta - eps * u)) / (2.0 * eps) * nv


def hessian_top_eig(grad_fn: GradFn, theta: np.ndarray, iters: int = 20, eps: float = 1e-3,
                    seed: int = 0, tol: float = 1e-4) -> Dict:
    """Power iteration for the eigenvalue of largest magnitude (sign kept). Returns value + convergence info."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(theta.shape)
    v /= np.linalg.norm(v)
    lam, hist = 0.0, []
    for _ in range(iters):
        hv = hvp(grad_fn, theta, v, eps)
        new = float(v @ hv)
        hist.append(new)
        nh = float(np.linalg.norm(hv))
        if nh == 0.0:
            break
        v = hv / nh
        if abs(new - lam) <= tol * max(abs(new), 1e-12):
            lam = new
            break
        lam = new
    return {"top_eigenvalue": lam, "iterations": len(hist), "history": hist,
            "converged": len(hist) < iters or (len(hist) > 1 and abs(hist[-1] - hist[-2]) <= tol * max(abs(hist[-1]), 1e-12))}


def hutchinson_trace(grad_fn: GradFn, theta: np.ndarray, n_probes: int = 10, eps: float = 1e-3, seed: int = 0) -> Dict:
    """trace(H) ~ mean_i z_i^T H z_i with Rademacher z. Returns mean and standard error over probes."""
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(n_probes):
        z = rng.choice([-1.0, 1.0], size=theta.shape)
        est.append(float(z @ hvp(grad_fn, theta, z, eps)))
    est = np.asarray(est)
    return {"trace": float(est.mean()), "std_err": float(est.std(ddof=1) / np.sqrt(len(est))) if len(est) > 1 else float("nan"),
            "n_probes": n_probes}


def roughness(loss_fn: LossFn, theta: np.ndarray, d: np.ndarray, span: float = 0.5, n: int = 101) -> Dict:
    """Loss along theta + t*d, t in [-span, span]; roughness = mean |second difference| / step^2 (0 for a straight line)."""
    t = np.linspace(-span, span, n)
    y = np.array([loss_fn(theta + a * d) for a in t])
    h = t[1] - t[0]
    sec = np.abs(y[2:] - 2 * y[1:-1] + y[:-2]) / (h * h)
    return {"t": t.tolist(), "loss": y.tolist(), "mean_abs_curvature": float(sec.mean()), "max_abs_curvature": float(sec.max())}
