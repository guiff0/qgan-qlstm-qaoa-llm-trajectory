# Combined change set (everything from this session, on top of the uploaded scripts.zip)

Apply by extracting this archive over your repo root (same layout as the original zip). 18 files differ
from the original: 5 new, 13 changed. `config/default_config.yaml.bak` is not included.

## 1. Environment rebuild
| File | Change |
|---|---|
| `rebuild_env.ps1` | Idempotent: reuses a healthy Python 3.11 venv (`-Force` to wipe); installs only missing / wrong-version packages; aborts on pip failure; verifies `dukascopy_python` (the module name is `dukascopy_python`, not `dukascopy`); runs `verify_environment --env-only`. |
| `scripts/check_requirements.py` (new) | Compares `requirements.txt` with what is installed; prints only unsatisfied specs. |
| `requirements.txt` | `numpy>=2.0.0` (the old `<2.0.0` pin silently downgraded NumPy and conflicted with PennyLane 0.44); removed the unused `tradedesk-dukascopy`. |
| `scripts/verify_environment.py` | `--env-only` flag; failures now print exception type and raising file:line. |

## 2. Data acquisition
| `scripts/acquire_all_data.py` | `dukascopy_python` raised `name 'requests' is not defined` (a bug inside the package). Added a direct Dukascopy datafeed client (daily `.bi5` candles) used automatically when the package is unusable. Not verified against the live servers. |

## 3. Repeated runs + entanglement sweep (resolves H1-H5 / M2 inconclusives with data, not looser rules)
| `scripts/run_repeated_experiment.py` | Validates model names, confirms a row was really appended, resumes from the results CSV (matched by run_id slug, since every ablation has `model_type == "QGAN-LLM"`), several models seed-major, `--dry-run`, `--status`, stops after 3 consecutive failures. |
| `scripts/gen_entanglement_sweep.py` (new) | Generates / applies (idempotent) six sweep ablations: 12 qubits, only topology and depth change. |
| `src/evaluation/entanglement_design.py` (new) | Level definitions; reads the level back from the `run_id` slug. |
| `config/default_config.yaml` | Contains the sweep block (applied) and the `forecast_path` keys in both GAN sections. |
| `scripts/run_hypothesis_tests.py` | H2/H5 use the controlled sweep when there are >= 12 runs over >= 3 levels (manipulation check + dose-response reported); M2 = designed level -> measured entropy -> ASR; otherwise the old pooled analysis, labelled `pooled_configs`. |
| `src/evaluation/verdicts.py` | Independent-run sweeps drop the "each point is a config" blocker; >= 30 runs, >= 3 per level and a passing manipulation check make H2/H5/M2 confirmatory (a clean null becomes `NOT_SUPPORTED`). |

## 4. Generator on the forecast path (makes H4, and H2/H5 via ASR, testable)
| `src/utils/generator_path.py` (new) | `forecast = linear([X, G(X)])`; ridge fit; fixed-seed evaluation subsample; persistence on the same rows. |
| `src/baselines/qgan_llm.py`, `classical_gan_llm.py` | `forecast_path: generator` mode (default in code stays `head_only`). Head fitted after GAN training; checkpoint is only "completed" after that. |
| `src/utils/gan_training.py` | Fingerprint ignores evaluation-only keys and, in `head_only` mode, the new keys, so existing checkpoints still resume. |
| `src/experiments/run_all.py` | Keeps a model-supplied `persistence_rmse`; `latency_scope` gains `quantum_circuit_in_path` / `classical_generator_in_path` for GAN models in generator mode. |
| `scripts/run_hypothesis_tests.py` (H4) | Compares only primary-config, generator-path rows of both models. |

## Tests
`tests/test_entanglement_sweep.py` (16) and `tests/test_generator_path.py` (new). Every pre-existing test module gives
identical results before and after this change set in the sandbox used (which has no torch, pennylane or statsmodels).
The torch tests in `test_generator_path.py` have NOT been executed anywhere: run `pytest tests -v` once.

## Order of operations
1. `.\rebuild_env.ps1`
2. Re-acquire data (`scripts.acquire_all_data`), then `scripts.prepare_data`.
3. **`python -m src.experiments.run_all`** -- always first among the results steps: it trains and evaluates every
   model and ablation (including the six sweep levels, seed 42) and is the only thing that writes
   `results/all_results.csv`. (`run_pipeline.py` does this as its `train_eval:*` steps.)
4. More seeds: `python -m scripts.run_repeated_experiment --model "Classical GAN-LLM,QGAN-LLM" --n-runs 8 --base-seed 43`
   and the sweep command printed by `python -m scripts.gen_entanglement_sweep` (each call runs `run_all --only ... --seed N`).
5. `python -m scripts.verify_chapter4`
6. `python -m scripts.run_hypothesis_tests`

Steps 5 and 6 now exit with code 2 and a "run run_all first" message if `results/all_results.csv` is missing or has
no rows (`src/utils/require_results.py`, `tests/test_require_results.py`).

## Known open items
H1 and H3 still pool QGAN rows across configs (H4 no longer does); the QLSTM forecaster's persistence RMSE is still
scored on different rows than its model RMSE; a true zero-entanglement sweep level would need a change to `circuits.py`.
