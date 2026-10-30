"""
Delta forecast target (src/evaluation/one_step_ahead.make_target):
  * pairing / alignment is right,
  * a head that predicts 0 scores EXACTLY persistence,
  * on a series with a level shift between train and test and lossy features (the situation behind
    "models are 4.5x worse than persistence"), a level-frame linear head is far worse than persistence
    while the delta-frame head is not.
numpy-only tests run anywhere; the torch ones are skipped when torch is missing.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.one_step_ahead import make_target, resolve_frame  # noqa: E402


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_alignment_delta_and_level():
    X = np.arange(10, dtype=float).reshape(5, 2)
    y = np.array([1.0, 1.5, 1.2, 2.0, 2.0])
    d = make_target(X, y, "delta")
    assert np.array_equal(d.X, X[:-1])
    assert np.allclose(d.target, [0.5, -0.3, 0.8, 0.0])
    assert np.allclose(d.y_next, y[1:]) and np.allclose(d.last, y[:-1])
    assert np.all(d.reference == 0)
    lv = make_target(X, y, "level")
    assert np.allclose(lv.target, y[1:]) and np.allclose(lv.reference, y[:-1])


def test_delta_precision_is_float64():
    y = (1.1 + np.arange(5) * 1e-5).astype(np.float64)
    d = make_target(np.zeros((5, 1)), y, "delta")
    assert np.allclose(d.target, 1e-5, atol=1e-12)


def test_zero_head_in_delta_frame_equals_persistence_rmse():
    rng = np.random.default_rng(0)
    y = 1.1 + np.cumsum(rng.normal(0, 1e-4, 5000))
    t = make_target(np.zeros((len(y), 3)), y, "delta")
    persistence = _rmse(y[1:], y[:-1])
    assert abs(_rmse(t.target, np.zeros_like(t.target)) - persistence) < 1e-15


def test_resolve_frame_validates():
    assert resolve_frame({}) == "delta" and resolve_frame({"forecast_target": "LEVEL"}) == "level"
    try:
        resolve_frame({"forecast_target": "nope"})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _lstsq(X, y):
    A = np.c_[X, np.ones(len(X))]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _pred(w, X):
    return np.c_[X, np.ones(len(X))] @ w


def test_level_head_loses_to_persistence_but_delta_head_does_not():
    rng = np.random.default_rng(1)
    n = 6000
    price = 1.10 + np.cumsum(rng.normal(0, 1e-4, 2 * n))
    price[n:] += 0.12                                     # regime/level shift into the "test" period
    feats = np.c_[price + rng.normal(0, 4e-4, 2 * n),     # lossy copy of the price (like a PCA'd feature)
                  rng.normal(size=(2 * n, 2))]
    tr, te = slice(0, n), slice(n, 2 * n)
    persistence = _rmse(price[te][1:], price[te][:-1])

    lv_tr, lv_te = make_target(feats[tr], price[tr], "level"), make_target(feats[te], price[te], "level")
    level_rmse = _rmse(lv_te.target, _pred(_lstsq(lv_tr.X, lv_tr.target), lv_te.X))

    d_tr, d_te = make_target(feats[tr], price[tr], "delta"), make_target(feats[te], price[te], "delta")
    delta_rmse = _rmse(d_te.target, _pred(_lstsq(d_tr.X, d_tr.target), d_te.X))

    assert level_rmse > 5 * persistence          # the failure mode (10-50x persistence across seeds)
    # The delta head is far better (~1-2x persistence across seeds) but NOT guaranteed <= persistence:
    # a feature that carries the price level still leaks the train->test level shift through its
    # fitted coefficient. Hence the loose bound, and why results must be read against persistence.
    assert delta_rmse < 3 * persistence and delta_rmse < level_rmse / 4


# ------------------------------------------------------------------ torch-dependent
def test_fingerprint_ignores_forecast_target():
    try:
        import torch  # noqa: F401
    except Exception:
        return
    from src.utils.gan_training import config_fingerprint
    a = config_fingerprint({"epochs": 3, "forecast_target": "delta"}, 42, "t")
    b = config_fingerprint({"epochs": 3, "forecast_target": "level"}, 42, "t")
    c = config_fingerprint({"epochs": 3}, 42, "t")
    d = config_fingerprint({"epochs": 4}, 42, "t")
    assert a == b == c and a != d


def test_refit_head_for_frame_rewrites_checkpoint(tmp_path=None):
    try:
        import torch
    except Exception:
        return
    import tempfile
    from src.utils.gan_training import refit_head_for_frame, save_checkpoint, load_checkpoint
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 4)).astype(np.float32)
    delta = (X @ np.array([1e-4, 0, 0, -2e-4]) + 3e-6).astype(np.float64)
    head = torch.nn.Linear(4, 1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.ckpt")
        ckpt = {"fingerprint": "fp", "forecast_head": head.state_dict(), "head_mse": 1.0}   # old: no frame key
        save_checkpoint(path, ckpt)
        mse = refit_head_for_frame(head, X, delta, ckpt, path, "delta")
        reloaded = load_checkpoint(path, "fp")
        assert reloaded["target_frame"] == "delta" and reloaded["head_mse"] == mse
        assert np.allclose(head.weight.detach().numpy().ravel(), [1e-4, 0, 0, -2e-4], atol=1e-6)
        assert os.path.isdir(os.path.join(d, "history"))          # the old-frame checkpoint was archived
