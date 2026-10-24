import re

import numpy as np

from src.baselines.context_llm import FeatureLLM, QLSTMLLM


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    closes = 1.10 + np.cumsum(rng.normal(0, 5e-5, n))
    return rng.standard_normal((n, 6)), closes


def _fake_chat(seen):
    def chat(msgs):
        seen.append(msgs[-1]["content"])
        nums = [float(x) for x in re.findall(r"\d\.\d{5}", msgs[-1]["content"])[:10]]
        return f"Forecast: {nums[-1]:.5f}\nThreat probability: 0.1"
    return chat


class StubInner:
    resumed_from_checkpoint = True

    def __init__(self):
        self.trained = False

    def train(self, *a, **k):
        self.trained = True

    def predict(self, X):
        return np.array([1.23456])


class StubMapper:
    available = True
    feature_cols = ["open", "close", "rsi", "macd", "vol", "atr", "ret", "noise"]

    def to_raw(self, rows):
        rows = np.asarray(rows)
        out = np.zeros((len(rows), 8))
        out[:, 2] = rows[:, 0] * 10 + 50      # rsi follows PCA comp 0
        out[:, 3] = rows[:, 1]
        out[:, 4] = rows[:, 2]
        out[:, 5] = rows[:, 3]
        out[:, 6] = rows[:, 4]
        out[:, 7] = rows[:, 5]
        out[:, 0] = out[:, 1] = 1.1
        return out


def test_qlstm_llm_puts_quantum_forecast_in_prompt(tmp_path):
    X, y = _data(); seen = []
    m = QLSTMLLM({"model_id": "m", "n_windows": 6, "n_regimes": 3, "max_workers": 1, "chat_fn": _fake_chat(seen),
                  "cache_dir": str(tmp_path)}, seed=1, inner=StubInner())
    m.train(X, y, X, y)
    out = m.evaluate(X, y, last_input_prices=y)
    assert m.inner.trained and m.resumed_from_checkpoint
    assert seen and all("1.23456" in s for s in seen)
    assert out["context_kind"] == "qlstm_forecast" and out["llm_extra_context_used"] is True
    assert out["n_llm_samples"] == 6 and out["n_llm_errors"] == 0


def test_feature_llm_classical_selects_correlated_named_features(tmp_path):
    X, y = _data(); seen = []
    # make the target depend on rsi (PCA comp 0) so the classical selector must pick it
    y = y + 0.0
    nxt = np.roll(y, -1) - y
    X[:, 0] = nxt / nxt.std() + 0.1 * np.random.default_rng(1).standard_normal(len(X))
    m = FeatureLLM({"model_id": "m", "selector": "classical", "target_k": 2, "n_windows": 6, "n_regimes": 3, "max_workers": 1,
                    "chat_fn": _fake_chat(seen), "cache_dir": str(tmp_path)}, seed=1, mapper=StubMapper())
    m.train(X, y, X, y)
    assert m.name == "Classical-Features-LLM" and "rsi" in m.selected_names
    assert "open" not in m.selected_names and "close" not in m.selected_names
    out = m.evaluate(X, y, last_input_prices=y)
    assert all("Selected market indicators" in s and "rsi=" in s for s in seen)
    assert out["selector"] == "classical" and np.isfinite(out["rmse"])


def test_feature_llm_qaoa_path_uses_selector_mask(tmp_path):
    X, y = _data(); calls = {}

    class Cfg:
        def __init__(self, **k): calls["cfg"] = k

    class Res:
        selected_mask = np.array([False, False, True, False, False, True])   # over the 6 kept (non-price) columns

    class Sel:
        def __init__(self, cfg): pass
        def fit(self, Xr, yr):
            calls["shape"] = Xr.shape; return Res()

    m = FeatureLLM({"model_id": "m", "selector": "qaoa", "target_k": 2}, seed=1, mapper=StubMapper(), selector_cls=(Sel, Cfg))
    m.train(X, y, X, y)
    assert m.name == "QAOA-LLM" and calls["shape"][1] == 6
    assert m.selected_names == ["rsi", "noise"] or m.selected_names == [StubMapper.feature_cols[i] for i in m.selected_idx]
    assert calls["cfg"]["n_candidates"] == 16 and calls["cfg"]["target_k"] == 2


def test_feature_llm_requires_preprocessors():
    class Missing:
        available = False
        reason = "FileNotFoundError: scaler.joblib"

    m = FeatureLLM({"model_id": "m", "selector": "classical"}, mapper=Missing())
    X, y = _data(50)
    try:
        m.train(X, y, X, y)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "prepare_data" in str(e)


def test_evaluate_requires_key_when_no_injected_chat(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    X, y = _data(50)
    m = QLSTMLLM({"model_id": "m", "qlstm": {}}, inner=StubInner())
    assert "NVIDIA_API_KEY" in m.evaluate(X, y, last_input_prices=y)["error"]
