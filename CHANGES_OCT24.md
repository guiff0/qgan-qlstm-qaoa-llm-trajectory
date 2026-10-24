# Changes of October 24, 2026: the LLM now produces the forecast and the threat probability

## Why
The Oct 21 run showed three things: the forecast came from a closed-form linear head that never saw the
generator; every LLM call failed with HTTP 410 ("meta/llama-3.3-70b-instruct has reached its end of life on
2026-08-26"); and DV2/DV3/DV5 came from a RandomForest, not from an LLM score. This change wires the LLM in,
as the proposal describes.

## What changed
| File | Change |
|---|---|
| `src/llm/llm_pipeline.py` (new) | One chat call per window returns `Forecast:` and `Threat probability:`. Retries on 429/5xx, raises `LLMModelUnavailableError` on 404/410, resolves the model id against `GET /models` before any work, caches replies on disk (`results/llm_cache`), builds price-space FGSM/PGD windows from a surrogate AR model, samples windows stratified by realised volatility, and computes `llm_rmse`, `llm_mae`, `llm_asr_*`, `llm_threat_fpr/accuracy/recall/f1/auprc`. |
| `src/llm/threat_scoring.py` | Unchanged, now actually called (parses the probability). |
| `src/baselines/qgan_llm.py`, `classical_gan_llm.py` | `evaluate()` ends with the LLM stage; the generator's nearest synthetic rows are mapped to closes and put in the prompt. `forecast_head` metrics stay as the closed-form head baseline (columns `rmse`, `asr`). |
| `src/baselines/llm_forecaster.py` | Now the control condition (same prompts, attacks and metrics, no synthetic context). Uses real closes, not `X[:, 0]` (a PCA component). Default `api_base` fixed (it pointed at `api.nvidia.com`, not `integrate.api.nvidia.com`). |
| `src/experiments/run_all.py` | Resolves the LLM model id before training and stops early if none is served; passes `llm_cfg` to every `evaluate()`. |
| `scripts/prepare_data.py` | Saves `scaler.joblib`, `pca.joblib`, `feature_cols.json` next to the arrays. **Re-run it once**; without these files the synthetic context is switched off and the result says so (`llm_synthetic_context_used = False`). |
| `scripts/run_hypothesis_tests.py` | H5 uses `llm_threat_accuracy` when 4+ configurations have it, else the stopgap detector, and records which. |
| `scripts/check_llm_endpoint.py` (new) | One-call check of key, model list, and reply format. Run it first. |
| `config/default_config.yaml` | `llm_forecaster:` replaced by `llm_pipeline:`. Default model `nvidia/llama-3.3-nemotron-super-49b-v1.5` with fallbacks. |
| `tests/test_llm_pipeline.py`, `tests/test_llm_forecaster.py` | 14 tests with injected fakes. No network. |

## Decisions that are NOT from the proposal (report them in the manuscript)
1. **Model.** Llama 3.3 70B Instruct is retired on the hosted API. The default is NVIDIA's Llama-3.3-based Nemotron Super 49B, resolved against the live model list. If you use another model, say so in Chapter 3.
2. **Sample, not full test set.** 102 windows per model (34 per volatility tercile), 2 live calls each. The proposal's 100 observations per regime cannot be met per regime at this size for regime-level tests; raise `n_windows` to 300 for that.
3. **Attacks.** FGSM/PGD against an LLM cannot use its gradients. Windows are perturbed in price space against an AR surrogate (epsilon = 2 x the std of 1-minute changes). FGSM and PGD differ only by a random start and the step schedule, because the surrogate is linear. Success = directional flip relative to the last real close, the definition already used in `src/attacks/adversarial.py`.
4. **Synthetic context.** The 5 synthetic rows nearest to the current row (PCA space) are mapped back to close prices and shown to the LLM. The control arm (LLM Forecaster) sees none. The difference between QGAN-LLM, Classical GAN-LLM and the control is the only place a generator effect can appear in the LLM columns.
5. **Detection labels.** Half the windows are sent clean (label 0), half attacked (label 1); threshold 0.70 (QACL table).

## Not verified
Nothing here has been run against the live NVIDIA endpoint (no key or route in the development sandbox). The tests prove the logic with fakes. Before a full run: `python scripts/check_llm_endpoint.py`.

## Run order
1. `python scripts/prepare_data.py` (saves the preprocessors; delete the pipeline-state entry for that step if `run_pipeline.py` skips it)
2. `set NVIDIA_API_KEY=...` then `python scripts/check_llm_endpoint.py`
3. `python run_pipeline.py` (the GAN models retrain only if their config fingerprint changed; the LLM stage adds roughly 204 calls per model)
