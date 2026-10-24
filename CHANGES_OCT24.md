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
1. **Model.** Llama 3.3 70B Instruct is retired, and the Oct 24 listing of the hosted API contains no Llama 3.3 model at all (it lists Llama 3.1 Nemotron 70B / 51B / Ultra 253B, Llama 3.2 vision models, and others). The default is now `nvidia/llama-3.1-nemotron-70b-instruct` (Llama 3.1 70B base, NVIDIA-tuned), with the 51B and Ultra 253B as fallbacks, resolved against the live model list. Chapter 3 must name the model actually used and say that it replaces Llama 3.3 70B.
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

## Parallel execution
`python run_pipeline.py --jobs 4` runs the train_eval steps four at a time and records them in the state file.
`.\scripts\launch_parallel.ps1 -MaxParallel 4` instead opens one PowerShell window per model (slowest first),
saves each window's output to `logs\parallel_<model>.txt`, then runs `verify_chapter4` and `run_hypothesis_tests`
when every window exits 0. It does not update `data\pipeline_state.json`.

## QLSTM-LLM, QAOA-LLM, Classical-Features-LLM (added Oct 24)
New file `src/baselines/context_llm.py`; registered in `run_all.py` only when `NVIDIA_API_KEY` is set.
| Model | What the LLM sees besides the last 10 real closes | Control |
|---|---|---|
| QLSTM-LLM | the QLSTM Forecaster's forecast for the real row (trained with the QLSTM's own checkpoint/resume) | LLM Forecaster (no quantum text) |
| QAOA-LLM | the values of 8 NAMED engineered features chosen by the QAOA QUBO selector | Classical-Features-LLM (top-8 by abs. correlation) |
| Classical-Features-LLM | the values of 8 named features chosen classically | (it is the control) |

Notes that belong in the manuscript:
- Named features are reconstructed from the 32 PCA components (approximate). Price levels (open, high, low, close) are excluded from selection. The selection target is the next-close change, not the price level.
- The QLSTM forecast is poor (RMSE 0.115, hundreds of times the persistence RMSE); expect the LLM to gain nothing from it.
- All three use the same 102 stratified windows, the same price-space attacks and the same threat scoring as the other LLM rows, so their rows are directly comparable. One seed: treat all differences as exploratory.
- `QAOA-Enhanced LLM` (head-based, with no LLM call) is unchanged and still reported under that name.

## Authentication check (added after the first live check)
`GET /models` is a public endpoint, so listing models and resolving the model id succeeded even with an invalid key,
and the first real call returned HTTP 401 ("Invalid JWT serialization: Missing dot delimiter(s)"). Now:
- `run_all.py` and `scripts/check_llm_endpoint.py` make one authenticated 1-token call right after resolving the model
  (`llm_pipeline.verify_auth`), before any training starts.
- HTTP 401/403 raises `LLMAuthError` with the fix; it is not retried and not counted as a per-window error.
- `get_api_key()` strips surrounding quotes/spaces and a pasted `Bearer ` prefix; `describe_key()` prints only the
  key's length and whether it starts with `nvapi-`.

## Model access check (added after the second live check)
`nvidia/llama-3.1-nemotron-70b-instruct` is in the public `/models` catalog but returned HTTP 404 ("Function ... Not found
for account ...") for this account: listed is not callable. `resolve_model_id` now takes a `probe_fn` and `run_all.py` passes a
real 1-token call, so a candidate is accepted only if this account can call it. `python scripts\check_llm_endpoint.py --probe-all`
tests every Llama-family chat model in the catalog and prints which ones work, so `llm_pipeline.model_id` can be set from evidence.

## Timeouts (added after the third live check)
The `--probe-all` run died on `ReadTimeout (60 s)`: `chat()` did not handle network errors. Now timeouts and connection errors are
retried with backoff and then raised as `RuntimeError("network error ...")`; `probe_model` waits 45 s with one retry and
reports "no answer ... cold start or overloaded" instead of crashing; `discover_models` probes 4 models at a time, prints each
result as it arrives, and contains per-model errors. `llm_pipeline.timeout` is 120 s for the real run.
