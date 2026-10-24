"""One-minute check before an LLM run: key set? model served? does one dual query parse?

    python scripts/check_llm_endpoint.py

Lists the Llama-family models the endpoint serves, resolves the configured model id (with
fallbacks), sends ONE prompt built from synthetic prices, and prints the raw reply and the
parsed forecast and threat probability. Costs one API call. Exits non-zero on any failure.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import llm_pipeline as lp  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    if not os.environ.get("NVIDIA_API_KEY"):
        print("NVIDIA_API_KEY is not set.")
        return 2
    cfg = load_config()["llm_pipeline"]
    base = cfg.get("api_base", lp.DEFAULT_API_BASE)
    models = lp.list_models(base)
    print(f"{len(models)} models served; Llama-family: {sorted(m for m in models if 'llama' in m.lower())[:15]}")
    try:
        model_id = lp.resolve_model_id(cfg["model_id"], cfg.get("fallback_model_ids", []), base)
    except lp.LLMModelUnavailableError as exc:
        print(exc)
        return 3
    print(f"model in use: {model_id}")
    closes = [1.10000 + 0.00002 * i for i in range(cfg.get("window_size", 10))]
    nothink = bool(cfg.get("nothink", False)) and "nemotron" in model_id.lower()
    raw = lp.chat(lp.build_dual_messages(closes, None, 1, nothink), model_id, base, max_tokens=int(cfg.get("max_tokens", 64)))
    parsed = lp.parse_dual_response(raw)
    print(f"raw reply: {raw!r}\nparsed:    {parsed}")
    ok = parsed["forecast"] is not None and parsed["threat_probability"] is not None
    print("OK: reply parses into a forecast and a threat probability" if ok else
          "FAIL: reply does not contain both 'Forecast:' and 'Threat probability:' lines; adjust llm_pipeline.nothink / max_tokens or the prompt")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
