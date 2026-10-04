import numpy as np

from src.baselines.llm_forecaster import LLMForecaster


def _toy_data(n=200, f=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float32)
    y = (X[:, 0] + 0.01 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def test_evaluate_with_mocked_query_fn_produces_metrics():
    X, y = _toy_data()

    def fake_query(recent_closes, model_id, horizon_minutes=1, api_base=None):
        return recent_closes[-1] + 0.001

    model = LLMForecaster({"max_llm_calls": 15, "query_fn": fake_query}, seed=1)
    model.build()
    model.train(None, None, None, None)
    result = model.evaluate(X, y)

    assert result["n_llm_samples"] == 15
    assert result["n_llm_errors"] == 0
    assert np.isfinite(result["rmse"])


def test_evaluate_caps_calls_at_max_llm_calls():
    X, y = _toy_data(n=500)

    calls = []

    def fake_query(recent_closes, model_id, horizon_minutes=1, api_base=None):
        calls.append(1)
        return recent_closes[-1]

    model = LLMForecaster({"max_llm_calls": 5, "query_fn": fake_query}, seed=1)
    model.evaluate(X, y)
    assert len(calls) == 5


def test_evaluate_handles_query_errors_without_crashing():
    X, y = _toy_data(n=100)

    def flaky_query(recent_closes, model_id, horizon_minutes=1, api_base=None):
        raise RuntimeError("simulated API failure")

    model = LLMForecaster({"max_llm_calls": 5, "query_fn": flaky_query}, seed=1)
    result = model.evaluate(X, y)
    assert result["n_llm_errors"] == 5
    assert result["n_llm_samples"] == 0
    assert np.isnan(result["rmse"])


def test_predict_raises_not_implemented():
    model = LLMForecaster(seed=1)
    try:
        model.predict(np.zeros((1, 8)))
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
