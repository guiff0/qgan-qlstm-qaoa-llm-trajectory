"""
Regression test for the same-row target leakage fix
(src/evaluation/one_step_ahead.py). Builds a synthetic random walk --
by construction unforecastable from same-row features, since each
step is independent noise -- the same test design the earlier audit
used to first catch the bug. If a same-row model can still beat the
one-step persistence floor here, it's reading the target out of its
own input again.
"""
import numpy as np
import pytest

from src.evaluation.one_step_ahead import shift_for_one_step_ahead
from src.utils.config import load_config, merge_override


def _random_walk_data(n=4000, f=32, seed=0):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.standard_normal(n)) * 0.01 + 100.0
    X = np.column_stack([y] + [y + 0.001 * rng.standard_normal(n) for _ in range(f - 1)]).astype(np.float32)
    return X, y.astype(np.float32)


def test_shift_for_one_step_ahead_basic():
    X = np.arange(10).reshape(10, 1).astype(np.float32)
    y = np.arange(100, 110).astype(np.float32)
    X2, y2 = shift_for_one_step_ahead(X, y)
    assert len(X2) == len(y2) == 9
    assert X2[0, 0] == 0  # X[0] now pairs with...
    assert y2[0] == 101   # ...y[1], not y[0]


def test_shift_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        shift_for_one_step_ahead(np.zeros((5, 2)), np.zeros(4))


def test_shift_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 2 rows"):
        shift_for_one_step_ahead(np.zeros((1, 2)), np.zeros(1))


@pytest.mark.parametrize("model_module,model_class,config_key", [
    ("src.baselines.classical_gan_llm", "ClassicalGANLLM", "classical_gan_llm"),
])
def test_same_row_model_cannot_beat_persistence_floor_on_random_walk(model_module, model_class, config_key):
    import importlib
    X, y = _random_walk_data()
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[:200], y[:200]
    X_test, y_test = X[split:], y[split:]

    cfg = load_config()
    c = merge_override(cfg[config_key], {"epochs": 2, "batch_size": 64})
    cls = getattr(importlib.import_module(model_module), model_class)
    model = cls(c, seed=1)
    model.train(X_train, y_train, X_val, y_val)
    result = model.evaluate(X_test, y_test)

    one_step_floor = float(np.sqrt(np.mean(np.diff(y_test.astype(np.float64)) ** 2)))
    assert result["rmse"] >= one_step_floor, (
        f"model RMSE ({result['rmse']}) beat the one-step persistence floor "
        f"({one_step_floor}) on an unforecastable random walk -- same-row "
        f"leakage has regressed."
    )
