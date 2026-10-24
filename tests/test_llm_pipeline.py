"""Tests for src/llm/llm_pipeline.py. All network access is injected; nothing here calls NVIDIA."""
import re

import numpy as np
import pytest

from src.llm import llm_pipeline as lp


class FakeResp:
    def __init__(self, status=200, text="", payload=None):
        self.status_code, self.text, self._p = status, text, payload

    def json(self):
        return self._p


def _ok(text):
    return FakeResp(200, "", {"choices": [{"message": {"content": text}}]})


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")


def _series(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return 1.10 + np.cumsum(rng.normal(0, 5e-5, n))


def _fake_chat(msgs):
    """Forecast = last close in the prompt; threat probability high when the window is jagged."""
    user = msgs[-1]["content"]
    nums = [float(x) for x in re.findall(r"\d\.\d{5}", user.split("A generative")[0])]
    jag = np.mean(np.abs(np.diff(nums, 2))) if len(nums) > 2 else 0
    return f"Forecast: {nums[-1]:.5f}\nThreat probability: {0.9 if jag > 1e-4 else 0.1}"


def test_parse_variants():
    assert lp.parse_dual_response("Forecast: 1.10234\nThreat probability: 0.83") == {"forecast": 1.10234, "threat_probability": 0.83}
    r = lp.parse_dual_response("<think>Forecast: 9</think>\nForecast: 1.1\nThreat probability: 0.2")
    assert r["forecast"] == 1.1 and r["threat_probability"] == 0.2
    assert lp.parse_dual_response("no numbers")["forecast"] is None
    assert lp.parse_dual_response("Forecast: 1.1")["threat_probability"] is None  # never defaulted to 0


def test_chat_retries_429_then_succeeds():
    seq = [FakeResp(429, "slow down"), _ok("Forecast: 1.1\nThreat probability: 0.1")]
    out = lp.chat([{"role": "user", "content": "x"}], "m", post=lambda *a, **k: seq.pop(0), sleep=lambda s: None)
    assert "Forecast" in out and not seq


def test_chat_410_raises_model_unavailable():
    with pytest.raises(lp.LLMModelUnavailableError):
        lp.chat([], "meta/llama-3.3-70b-instruct", post=lambda *a, **k: FakeResp(410, "end of life"))


def test_chat_requires_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY")
    with pytest.raises(Exception):
        lp.chat([], "m", post=lambda *a, **k: _ok("x"))


def test_resolve_model_id_fallback_and_error():
    avail = lambda base: ["nvidia/llama-3.3-nemotron-super-49b-v1.5", "other/model"]  # noqa: E731
    assert lp.resolve_model_id("meta/llama-3.3-70b-instruct", ["nvidia/llama-3.3-nemotron-super-49b-v1.5"], list_fn=avail) \
        == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    with pytest.raises(lp.LLMModelUnavailableError) as e:
        lp.resolve_model_id("gone/model", [], list_fn=avail)
    assert "nemotron" in str(e.value)


def test_attack_window_flips_surrogate_and_respects_eps():
    c = _series()
    coef, _ = lp.fit_ar_surrogate(c, 10)
    rng = np.random.default_rng(0)
    w = c[100:110]; last = c[109]; eps = 2e-4
    f0 = coef @ (w - w.mean()) + w.mean()
    for kind in ("fgsm", "pgd"):
        adv = lp.attack_window(w, last, coef, kind, eps, eps * 0.1, 10, rng)
        assert np.max(np.abs(adv - w)) <= eps + 1e-12
        f1 = coef @ (adv - adv.mean()) + adv.mean()
        assert abs(f1 - f0) > 0  # moved
        assert np.sign(f0 - last) != np.sign(f1 - last) or abs(f1 - last) < abs(f0 - last)  # toward / across the last close


def test_stratified_indices_balanced_and_valid():
    c = _series(5000)
    idx, counts = lp.stratified_indices(c, 10, 99, 3, seed=1)
    assert counts == [33, 33, 33] and len(idx) == 99
    assert idx.min() >= 10 and idx.max() < len(c) - 1


def test_end_to_end_with_fake_chat(tmp_path):
    c = _series(); y_next = c[1:]; closes = c[:-1]
    X = np.random.default_rng(0).normal(size=(len(closes), 4))
    cfg = {"model_id": "m", "window_size": 10, "n_windows": 30, "n_regimes": 3, "max_workers": 2}
    res = lp.evaluate_llm_pipeline(closes, y_next, X, cfg, seed=3, chat_fn=_fake_chat, cache_dir=str(tmp_path))
    assert res["llm_n_calls"] == 60 and res["llm_n_errors"] == 0 and res["llm_parse_failures"] == 0
    assert np.isfinite(res["llm_rmse"]) and np.isfinite(res["llm_threat_accuracy"])
    assert res["llm_synthetic_context_used"] is False
    # cache: a second run makes no new calls
    res2 = lp.evaluate_llm_pipeline(closes, y_next, X, cfg, seed=3, chat_fn=lambda m: (_ for _ in ()).throw(AssertionError("no call")),
                                    cache_dir=str(tmp_path))
    assert res2["llm_n_cached"] == 60 and res2["llm_rmse"] == res["llm_rmse"]


def test_errors_are_counted_not_swallowed(tmp_path):
    c = _series(); closes, y_next = c[:-1], c[1:]
    cfg = {"model_id": "m", "window_size": 10, "n_windows": 9, "n_regimes": 3}

    def boom(msgs):
        raise RuntimeError("boom")

    res = lp.evaluate_llm_pipeline(closes, y_next, None, cfg, chat_fn=boom, cache_dir=str(tmp_path))
    assert res["llm_n_errors"] == 18 and np.isnan(res["llm_rmse"]) and res["llm_n_forecasts"] == 0


def test_model_unavailable_propagates(tmp_path):
    c = _series(); closes, y_next = c[:-1], c[1:]

    def gone(msgs):
        raise lp.LLMModelUnavailableError("410")

    with pytest.raises(lp.LLMModelUnavailableError):
        lp.evaluate_llm_pipeline(closes, y_next, None, {"model_id": "m", "n_windows": 6}, chat_fn=gone, cache_dir=str(tmp_path))


class StubMapper:
    available = True

    def to_close(self, rows):
        return 1.10 + 0.0 * rows[:, 0] + rows[:, 0] * 1e-4


def test_synthetic_context_reaches_the_prompt(tmp_path):
    c = _series(); closes, y_next = c[:-1], c[1:]
    X = np.random.default_rng(0).normal(size=(len(closes), 4)); seen = []

    def spy(msgs):
        seen.append(msgs[-1]["content"]); return _fake_chat(msgs)

    gen = lambda n: np.random.default_rng(1).normal(size=(n, 4))  # noqa: E731
    cfg = {"model_id": "m", "window_size": 10, "n_windows": 6, "n_regimes": 3, "max_workers": 1}
    res = lp.evaluate_llm_pipeline(closes, y_next, X, cfg, generator_fn=gen, mapper=StubMapper(), chat_fn=spy, cache_dir=str(tmp_path))
    assert res["llm_synthetic_context_used"] is True and all("synthetic market states" in s for s in seen)


def test_evaluate_if_enabled_paths(monkeypatch):
    c = _series(); args = (c[:-1], c[1:], None, 0)
    assert lp.evaluate_if_enabled(None, *args) == {}
    assert lp.evaluate_if_enabled({"enabled": True}, None, c[1:], None, 0)["llm_pipeline_status"].startswith("skipped")
    monkeypatch.delenv("NVIDIA_API_KEY")
    assert "NVIDIA_API_KEY" in lp.evaluate_if_enabled({"enabled": True}, *args)["llm_pipeline_status"]
