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


def test_reasoning_prefix_per_model():
    assert lp.reasoning_prefix("nvidia/llama-3.3-nemotron-super-49b-v1.5") == "/no_think\n"
    assert lp.reasoning_prefix("nvidia/llama-3.1-nemotron-ultra-253b-v1") == "detailed thinking off\n"
    assert lp.reasoning_prefix("nvidia/llama-3.1-nemotron-70b-instruct") == ""
    msgs = lp.build_dual_messages([1.1] * 10, None, 1, prefix="/no_think\n")
    assert msgs[0]["content"].startswith("/no_think")


def test_key_sanitising_and_description(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", ' "nvapi-abc123" ')
    assert lp.get_api_key() == "nvapi-abc123"
    monkeypatch.setenv("NVIDIA_API_KEY", "Bearer nvapi-abc123")
    assert lp.get_api_key() == "nvapi-abc123"
    d = lp.describe_key()
    assert "nvapi-abc123" not in d and "length 12" in d and "removed" in d
    monkeypatch.setenv("NVIDIA_API_KEY", "your-key-here")
    assert "does NOT start with 'nvapi-'" in lp.describe_key()


def test_401_raises_auth_error_with_fix_and_propagates(tmp_path):
    bad = FakeResp(401, '{"detail":"Invalid JWT serialization"}')
    with pytest.raises(lp.LLMAuthError) as e:
        lp.chat([], "m", post=lambda *a, **k: bad)
    assert "nvapi-" in str(e.value) and "without quotes" in str(e.value).lower()
    with pytest.raises(lp.LLMAuthError):
        lp.verify_auth("m", post=lambda *a, **k: bad)
    c = _series(); closes, y_next = c[:-1], c[1:]

    def reject(msgs):
        raise lp.LLMAuthError("401")

    with pytest.raises(lp.LLMAuthError):
        lp.evaluate_llm_pipeline(closes, y_next, None, {"model_id": "m", "n_windows": 6}, chat_fn=reject, cache_dir=str(tmp_path))


def test_evaluate_if_enabled_reports_rejected_key(monkeypatch):
    c = _series()
    monkeypatch.setattr(lp, "evaluate_llm_pipeline", lambda *a, **k: (_ for _ in ()).throw(lp.LLMAuthError("401")))
    out = lp.evaluate_if_enabled({"enabled": True, "model_id": "m"}, c[:-1], c[1:], None, 0)
    assert out["llm_pipeline_status"].startswith("key rejected")


def _post_by_model(table):
    """fake requests.post: status per model id"""
    def post(url, headers=None, json=None, timeout=None):
        status = table.get(json["model"], 200)
        if status == 200:
            return _ok("pong")
        if status == 404:
            return FakeResp(404, '{"detail":"Function \'x\': Not found for account \'y\'"}')
        return FakeResp(status, "err")
    return post


def test_probe_model_distinguishes_missing_rate_limited_and_ok():
    post = _post_by_model({"a": 200, "b": 404, "c": 429})
    assert lp.probe_model("a", post=post)[0] is True
    ok, detail = lp.probe_model("b", post=post)
    assert ok is False and "Not found for account" in detail
    assert lp.probe_model("c", post=post, sleep=lambda s: None)[0] is True
    with pytest.raises(lp.LLMAuthError):
        lp.probe_model("d", post=_post_by_model({"d": 401}))


def test_resolve_skips_listed_but_uncallable_models():
    post = _post_by_model({"p": 404, "f1": 404, "f2": 200})
    avail = lambda base: ["p", "f1", "f2"]  # noqa: E731
    probe = lambda m: lp.probe_model(m, post=post)  # noqa: E731
    assert lp.resolve_model_id("p", ["f1", "f2"], list_fn=avail, probe_fn=probe) == "f2"
    with pytest.raises(lp.LLMModelUnavailableError) as e:
        lp.resolve_model_id("p", ["f1"], list_fn=avail, probe_fn=probe)
    msg = str(e.value)
    assert "listed but not callable" in msg and "--probe-all" in msg


def test_discover_models_filters_non_chat_and_reports_status():
    avail = lambda base: ["meta/llama-3.2-90b-vision-instruct", "nvidia/llama-3.2-nv-embedqa-1b-v1",  # noqa: E731
                          "meta/llama-guard-4-12b", "nvidia/llama-3.1-nemotron-70b-instruct", "other/model"]
    rows = lp.discover_models(list_fn=avail, probe_fn=lambda m: (m.endswith("instruct"), "x"))
    names = [m for m, _ in rows]
    assert names == ["meta/llama-3.2-90b-vision-instruct", "nvidia/llama-3.1-nemotron-70b-instruct"]


def test_chat_retries_timeouts_then_raises_runtime_error():
    import requests
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("slow")
        return _ok("pong")

    assert lp.chat([], "m", post=flaky, sleep=lambda s: None, retries=2) == "pong" and calls["n"] == 2

    def dead(*a, **k):
        raise requests.exceptions.ConnectionError("down")

    with pytest.raises(RuntimeError) as e:
        lp.chat([], "m", post=dead, sleep=lambda s: None, retries=1)
    assert "network error" in str(e.value)


def test_probe_reports_timeout_instead_of_crashing():
    import requests

    def dead(*a, **k):
        raise requests.exceptions.ReadTimeout("slow")

    ok, detail = lp.probe_model("m", post=dead, sleep=lambda s: None)
    assert ok is False and "cold start" in detail


def test_discover_contains_per_model_errors_and_reports_progress():
    seen = []

    def probe(m):
        if m.endswith("boom"):
            raise ValueError("kaput")
        return (True, "ok")

    rows = lp.discover_models(list_fn=lambda b: ["a/llama-x", "a/llama-boom"], probe_fn=probe, on_result=lambda m, r: seen.append(m))
    d = dict(rows)
    assert d["a/llama-x"][0] is True and d["a/llama-boom"][0] is False and "kaput" in d["a/llama-boom"][1]
    assert sorted(seen) == ["a/llama-boom", "a/llama-x"]
