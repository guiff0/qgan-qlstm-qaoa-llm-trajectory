"""
Tests src/llm/llm_inference.py's logic (prompt building, response
parsing, error handling) against a MOCKED HTTP response -- this
sandbox has no NVIDIA_API_KEY and no network route to api.nvidia.com
(see the module's own docstring), so these tests cannot and do not
verify the live API's actual behavior. They verify this code's own
logic is correct GIVEN a response of the documented shape.
"""
import pytest

from src.llm.llm_inference import (
    build_forecast_prompt, parse_forecast_response, query_llm_forecast,
)
from src.llm.nvidia_finetune import NvidiaApiKeyMissingError


def test_build_forecast_prompt_includes_all_closes_and_horizon():
    prompt = build_forecast_prompt([1.1000, 1.1005, 1.1010], horizon_minutes=5)
    assert "1.10000" in prompt
    assert "1.10050" in prompt
    assert "1.10100" in prompt
    assert "5 minute(s)" in prompt


def test_parse_forecast_response_extracts_number():
    assert parse_forecast_response("1.10234") == pytest.approx(1.10234)
    assert parse_forecast_response("The forecast is 1.1023.") == pytest.approx(1.1023)
    assert parse_forecast_response("-0.5") == pytest.approx(-0.5)


def test_parse_forecast_response_returns_none_for_unparseable_text():
    assert parse_forecast_response("I cannot predict that.") is None
    assert parse_forecast_response("") is None


def test_query_llm_forecast_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(NvidiaApiKeyMissingError):
        query_llm_forecast([1.1, 1.1001], model_id="fake-model")


def test_query_llm_forecast_parses_mocked_openai_shaped_response(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-key-for-testing")

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "1.10456"}}]}
        text = ""

    import src.llm.llm_inference as mod
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: FakeResponse())

    result = query_llm_forecast([1.1, 1.1005, 1.101], model_id="fake-model")
    assert result == pytest.approx(1.10456)


def test_query_llm_forecast_raises_on_error_status(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-key-for-testing")

    class FakeResponse:
        status_code = 401
        text = "invalid api key"

    import src.llm.llm_inference as mod
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: FakeResponse())

    with pytest.raises(RuntimeError, match="401"):
        query_llm_forecast([1.1, 1.1005], model_id="fake-model")


def test_query_llm_forecast_raises_on_unparseable_reply(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-key-for-testing")

    class FakeResponse:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": "I don't know."}}]}

    import src.llm.llm_inference as mod
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: FakeResponse())

    with pytest.raises(RuntimeError, match="Could not parse"):
        query_llm_forecast([1.1, 1.1005], model_id="fake-model")
