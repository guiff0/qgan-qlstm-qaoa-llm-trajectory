# Changes Oct 19, 2026 — why no quantum advantage is visible

Not run here: torch and PennyLane are unavailable in the authoring sandbox, so the torch edits
(`classical_gan_llm.py`, `qgan_llm.py`) were only byte-compiled. Everything numpy-only was executed
(`tests/test_advantage_diagnostics.py`, `python -m scripts.diagnose_no_advantage`).

## Findings (evidence: results/no_advantage_diagnosis.json)
| # | Cause | Evidence |
|---|---|---|
| C1 | Forecast head is fitted on real rows only and is the only inference module; generator never reaches it | 11 GAN-run RMSEs identical (0.000612877); head loss constant 3.7e-8 |
| C2 | Generator output = Linear(<Z>_n) so rank <= n_qubits (20 or 12) of 32 features | synth.npy rank 12 / 20 / 20 / 20; effective rank 6.0-9.9 |
| C3 | MSE-to-real-batch weight 10.0 (QGAN) vs 0.1 (classical): 100x; that term shrinks generator variance | synth per-feature SD 0.05-0.24 |
| C4 | Capacity unmatched: 87,072 (LSTM) vs 912 (20q) / 560 (12q) generator parameters | `lstm_generator_params`, `qgan_generator_params` |
| C5 | GAN training touched <= 9,600 of 3,715,921 rows (0.26%) | config max_batches_per_epoch=20 |
| C6 | One seed, no replicates | seed column has one value |
| C7 | Circuits are classically simulated: no runtime advantage possible; persistence beats every model | val persistence RMSE 1.64e-4 |

## Code
- `src/evaluation/advantage_diagnostics.py` (new): degenerate-group detector, rank, budget, capacity, skill-vs-persistence, Diebold-Mariano.
- `scripts/diagnose_no_advantage.py` (new): produces the table above from existing artifacts.
- `src/quantum/feature_map.py` (new): numpy statevector (matches a dense Kronecker reference to 1e-15) + matched classical random-feature control.
- `src/baselines/feature_map_forecaster.py`, `scripts/run_feature_map_experiment.py` (new): linear / classical_rf / quantum arms, circuit on the inference path, residual-over-persistence target, ridge in closed form, multi-seed, paired DM tests. **This experiment has not been run on the real data.**
- `classical_gan_llm.py`: `mse_weight` config (default 0.1). `qgan_llm.py` and `config/default_config.yaml`: `alpha` 10.0 -> 0.1 (now equal). Fingerprints change, so GAN checkpoints retrain.
- `scripts/run_hypothesis_tests.py`: H4 reports `status: degenerate` instead of a NaN t-statistic.

## Still open (not fixed)
Readout richer than <Z> (e.g. add ZZ) to lift the rank bound; replace the unpaired MSE term by moment matching; full-scale or longer-budget GAN training; seeds >= 5 per configuration.
