import re

import numpy as np

from src.baselines.llm_forecaster import LLMForecaster


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    closes = 1.10 + np.cumsum(rng.normal(0, 5e-5, n))
    X = rng.standard_normal((n, 8)).astype(np.float32)
    return X, closes.astype(np.float64)


def test_prompt_uses_real_closes_not_pca_component(tmp_path):
    X, y = _data(); seen = []

    def chat_fn(msgs):
        seen.append(msgs[-1]["content"]); return "Forecast: 1.10000\nThreat probability: 0.1"

    cfg = {"model_id": "m", "n_windows": 6, "n_regimes": 3, "max_workers": 1, "chat_fn": chat_fn, "cache_dir": None}
    model = LLMForecaster(cfg, seed=1)
    import src.llm.llm_pipeline as lp
    orig = lp.evaluate_llm_pipeline
    lp.evaluate_llm_pipeline = lambda *a, **k: orig(*a, **{**k, "cache_dir": str(tmp_path)})
    try:
        out = model.evaluate(X, y, last_input_prices=y)
    finally:
        lp.evaluate_llm_pipeline = orig
    nums = [float(v) for s in seen for v in re.findall(r"\d\.\d{5}", s)]
    assert nums and all(1.0 < v < 1.2 for v in nums)          # prices, never PCA components
    assert out["n_llm_samples"] == 6 and out["n_llm_errors"] == 0 and np.isfinite(out["rmse"])
    assert out["llm_synthetic_context_used"] is False         # control condition


def test_requires_real_close_series():
    X, y = _data(50)
    assert "error" in LLMForecaster({"model_id": "m"}, seed=1).evaluate(X, y)


def test_predict_raises_not_implemented():
    model = LLMForecaster(seed=1)
    try:
        model.predict(np.zeros((1, 8)))
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
