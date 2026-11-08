# Addendum to the pre-registration (7 Nov 2026)

`PREREGISTRATION.md` and `results/preregistration_*.json` are NOT edited. This addendum lists analyses and safeguards
added after the freeze. **None changes a margin, alpha, a seed list, or a decision rule**; `tests/test_claim3_claim5_analysis.py
::test_frozen_margins_and_alpha_are_unchanged` fails if any of them moves. Files of the frozen set that were edited are
logged with `python -m scripts.freeze_preregistration --record-deviation "<reason>"` (`results/deviations_*.json`).

| Addition | Why | Effect on the frozen rules |
|---|---|---|
| Experiment id (`src/utils/experiment.py`, `scripts/experiments.py`) | two 80-run studies must never be mixed or de-duplicated into each other | none (bookkeeping) |
| Balance audit (`scripts/check_balance.py`) | the quantum-vs-matched comparison is checked, not assumed (shared hyper-parameters, equal loss weight, equal rows seen, parameters within 5%, equal input dimensions, no per-arm tuning) | none; a failing audit stops a launch |
| Paired-by-seed tests (`src/evaluation/paired_tests.py`) | same seed = same scored rows, so paired differences are the natural secondary analysis; adds the 8-of-10 consistency count at run level | secondary only; the verdict still comes from the frozen rules |
| Row-level paired Diebold-Mariano rule (`scripts/run_temporal_validation.py`) | the frozen advantage rule names a paired DM test in >= 8 of 10 seeds; it needs per-row forecasts, saved from the first run that uses the new code (`results/predictions/`) | implements the frozen rule; reading used: p < 0.05 in the quantum arm's favour in >= 8 of the shared seeds AND RMSE/persistence < 1 |
| Decidability gates (`src/evaluation/decidability.py`, verdict NOT_DECIDABLE, rule R9) | an outcome that never moves (H3) or a detector at chance (H5) cannot answer the hypothesis; 'equivalence' on it is vacuous | reports H3/H5 as not decidable instead of a vacuous equivalence; thresholds: outcome range >= 20% of the margin, detector above chance at alpha 0.05 |
| Analysis plan (`config/analysis_plan.yaml`) | independent variable, outcome and scale, unit and replication, contrast and assumptions, fallback, for every hypothesis | restates the frozen design; `planned_vs_ran` is written to every results file |
| Temporal stability (`src/evaluation/temporal_validation.py`) | persistence skill by test block and VIX regime with a block-bootstrap CI | descriptive; within one test period, not rolling-origin validation |
| Preprocessing sensitivity (`src/data/variants.py`, `scripts/run_sensitivity_study.py`, `scripts/compare_sensitivity.py`) | winsorised volume, robust scaling, no interpolation, the documented rolling-moment features | robustness check on a cheap proxy (12-qubit QGAN vs matched classical, 5 seeds); not a re-test of H1/H4 |

Not changed and still open: the R3 verdict label NOT_SUPPORTED is issued at an adequate n without requiring equivalence,
while the pre-registration says an inconclusive result is never a trend; aligning that label with the text would change a
frozen rule and needs its own deviation entry.
