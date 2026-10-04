"""
LLM-based forecasting inference via NVIDIA's API -- the piece that was
missing even after nvidia_finetune.py existed: that module only
submits/polls FINE-TUNING jobs, it never actually queries a model for
a prediction. Without this, "an LLM forecasts" had no code path at all,
fine-tuned or not.

CANNOT BE VERIFIED FROM THIS SESSION: this sandbox has no
NVIDIA_API_KEY and no network route to api.nvidia.com (its egress
allowlist is a fixed set of package/doc hosts — see the system
prompt's network_configuration). Every function below is written
against NVIDIA's standard OpenAI-compatible chat-completions shape
(the common pattern across NIM-hosted models) and unit-tested with a
mocked HTTP response, but the actual live call has never been run
successfully against a real endpoint. Verify the request/response
shape against NVIDIA's current documentation and your actual model's
behavior before trusting any number this produces -- same caution
nvidia_finetune.py's own docstring already gives for the fine-tuning
endpoint.

Matches nvidia_finetune.py's established philosophy: no API key means
a clear, loud error, never a silently fabricated forecast.
"""
from __future__ import annotations

import re

import requests

from .nvidia_finetune import NvidiaApiKeyMissingError, _get_api_key


def build_forecast_prompt(recent_closes: list[float], horizon_minutes: int = 1) -> str:
    """
    Turns a short window of recent EUR/USD closes into a text prompt
    asking for a point forecast. Deliberately minimal (closes only, no
    full feature vector) -- an LLM prompt is not the place to paste all
    32 PCA components; this gives the model the same kind of signal a
    human analyst glancing at a price chart would have.
    """
    history = ", ".join(f"{c:.5f}" for c in recent_closes)
    return (
        f"You are forecasting the EUR/USD exchange rate. "
        f"The last {len(recent_closes)} 1-minute closing prices were: {history}. "
        f"Predict the closing price {horizon_minutes} minute(s) ahead. "
        f"Respond with ONLY the predicted number, nothing else."
    )


def parse_forecast_response(text: str) -> float | None:
    """Extracts the first number (int/float, optional sign) from the
    LLM's response text. Returns None (not 0.0 or a guess) if nothing
    parseable is found -- an unparseable response is a real failure
    the caller needs to see, not a silent zero."""
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def query_llm_forecast(recent_closes: list[float], model_id: str, horizon_minutes: int = 1,
                        api_base: str = "https://api.nvidia.com/v1", timeout: int = 30) -> float:
    """
    Queries model_id for a one-step-ahead forecast. Raises
    NvidiaApiKeyMissingError if NVIDIA_API_KEY isn't set (never falls
    back to a fabricated number), and RuntimeError with the API's own
    error text on any non-2xx response or an unparseable reply.
    """
    api_key = _get_api_key()
    prompt = build_forecast_prompt(recent_closes, horizon_minutes)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # deterministic point forecast, not creative sampling
        "max_tokens": 32,
    }

    response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(
            f"NVIDIA inference API returned {response.status_code}: {response.text}\n"
            f"Check your API key, model_id={model_id!r}, and that it's actually deployed "
            f"for inference (not just fine-tuning) on your account."
        )

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected NVIDIA API response shape: {data!r}\n"
            f"This function assumes an OpenAI-compatible chat-completions response -- "
            f"verify that against your actual model's real response before trusting this."
        ) from exc

    forecast = parse_forecast_response(content)
    if forecast is None:
        raise RuntimeError(f"Could not parse a numeric forecast out of the LLM's response: {content!r}")
    return forecast
