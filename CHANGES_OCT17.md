# Changes after the Oct 15-17 pipeline run

| File | Change | Why (evidence in the run log) |
|---|---|---|
| src/llm/llm_inference.py, config/default_config.yaml | api_base -> https://integrate.api.nvidia.com/v1 | All 20 LLM requests failed: certificate not valid for api.nvidia.com. Untested against a live endpoint. |
| run_pipeline.py | Completion check matches the run_id slug, not model_type | The 3 QGAN ablation configs were reported failed and re-run every time (all are model_type "QGAN-LLM"). |
| src/baselines/classical_lstm.py | ASR "last price" = unshifted close (last_input_prices), not PCA component 0 | LSTM ASR 0.0% was scored against a non-price column. Re-evaluate the saved checkpoint. |
| src/quantum/encoding_fidelity.py | Haar reference normalised in log space | M1_EFI was NaN at 20 qubits (0/0 underflow). |
| scripts/run_hypothesis_tests.py | One row per config (--no-dedupe to disable); H3 groups by noise_strength; mediation topology from run_id; ASR rows without asr_baseline are used only if the column is absent | Repeated evaluations of one checkpoint were counted as independent runs; H3 found 0 observations. |
| scripts/verify_chapter4.py, src/experiments/results_collector.py | Keyed by configuration, so "QGAN-LLM (12-qubit)" is found | Previously "no run found". ASR is in percent. |
| src/experiments/run_all.py | Logs persistence_rmse, rmse_over_persistence, asr_baseline | Persistence (val RMSE 1.64e-4) beat every model. |
| src/utils/run_identity.py, tests/test_run_identity.py | New helper and tests | |

Not run here: torch/pennylane are unavailable in this sandbox, so only the pure-Python helpers were executed. Re-run the LSTM evaluation and `python run_pipeline.py` locally.
