"""Check the LLM endpoint before an expensive run.

    python scripts/check_llm_endpoint.py              # key shape, model resolution, one dual query
    python scripts/check_llm_endpoint.py --probe-all  # which Llama-family chat models can THIS account call?

Costs one API call per candidate model (--probe-all: up to ~40 one-token calls). Exits non-zero on failure.
Note: GET /models is public and lists the whole catalog, including models your account cannot call (HTTP 404
"Function ... Not found for account"), so a real call is the only proof a model is usable.
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
    print(f"API key: {lp.describe_key()}")
    cfg = load_config()["llm_pipeline"]
    base = cfg.get("api_base", lp.DEFAULT_API_BASE)
    models = lp.list_models(base)
    print(f"{len(models)} models in the catalog (public list; says nothing about your account's access)")
    try:
        if "--probe-all" in sys.argv:
            print("Probing Llama-family chat models with one 1-token call each...")
            def show(m, res):
                good, detail = res
                print(f"  {'OK  ' if good else 'FAIL'} {m}" + ("" if good else f"   {detail[:110]}"), flush=True)
            rows = lp.discover_models(base, on_result=show)
            ok = [m for m, (good, _) in rows if good]
            if not ok:
                print("No Llama-family chat model is callable with this key. Check the account at build.nvidia.com.")
                return 3
            print(f"\nCallable now: {ok}\nSet llm_pipeline.model_id (and fallback_model_ids) in config/default_config.yaml.")
            return 0
        model_id = lp.resolve_model_id(cfg["model_id"], cfg.get("fallback_model_ids", []), base,
                                       probe_fn=lambda m: lp.probe_model(m, base))
    except lp.LLMAuthError as exc:
        print(exc)
        return 2
    except lp.LLMModelUnavailableError as exc:
        print(exc)
        return 3
    print(f"model in use (listed and callable): {model_id}")
    closes = [1.10000 + 0.00002 * i for i in range(cfg.get("window_size", 10))]
    prefix = lp.reasoning_prefix(model_id) if cfg.get("nothink", True) else ""
    raw = lp.chat(lp.build_dual_messages(closes, None, 1, prefix), model_id, base, max_tokens=int(cfg.get("max_tokens", 64)))
    parsed = lp.parse_dual_response(raw)
    print(f"raw reply: {raw!r}\nparsed:    {parsed}")
    ok = parsed["forecast"] is not None and parsed["threat_probability"] is not None
    print("OK: reply parses into a forecast and a threat probability" if ok else
          "FAIL: reply does not contain both 'Forecast:' and 'Threat probability:' lines; adjust llm_pipeline.nothink / max_tokens or the prompt")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
