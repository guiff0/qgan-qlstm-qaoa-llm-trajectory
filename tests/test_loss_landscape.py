"""Analytic checks for src/evaluation/loss_landscape.py (no torch/pennylane needed)."""
import numpy as np

from src.evaluation import loss_landscape as ll


def _quad(H):
    H = np.asarray(H, float)
    return (lambda t: 0.5 * float(t @ H @ t)), (lambda t: H @ t)


def test_top_eigenvalue_matches_numpy():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 8)); H = A @ A.T
    f, g = _quad(H)
    r = ll.hessian_top_eig(g, rng.standard_normal(8), iters=300, tol=1e-10)
    assert abs(r["top_eigenvalue"] - np.linalg.eigvalsh(H).max()) / np.linalg.eigvalsh(H).max() < 1e-3


def test_hutchinson_trace_close():
    H = np.diag(np.arange(1.0, 9.0)); f, g = _quad(H)
    r = ll.hutchinson_trace(g, np.zeros(8), n_probes=200, seed=3)
    assert abs(r["trace"] - np.trace(H)) < 3 * r["std_err"] + 1e-6


def test_slice_center_and_shape():
    f, g = _quad(np.eye(4)); th = np.ones(4)
    rng = np.random.default_rng(0)
    d1, d2 = ll.random_direction(th, rng), ll.random_direction(th, rng)
    s = ll.slice_2d(f, th, d1, d2, span=0.2, n=5)
    assert np.asarray(s["loss"]).shape == (5, 5)
    assert abs(np.asarray(s["loss"])[2, 2] - f(th)) < 1e-12 and abs(s["center_loss"] - f(th)) < 1e-12


def test_roughness_zero_for_linear_and_positive_for_quadratic():
    th = np.zeros(3); d = np.ones(3)
    assert ll.roughness(lambda t: float(t.sum()), th, d)["mean_abs_curvature"] < 1e-8
    assert ll.roughness(lambda t: float(t @ t), th, d)["mean_abs_curvature"] > 1.0
