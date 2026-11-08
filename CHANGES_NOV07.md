# 2026-11-07 — experiment id, reviewer claims 2-7

| Claim | What changed | Where |
|---|---|---|
| (new) experiment id | one id groups all run ids of a study; per-experiment state, logs, manifests; analysis refuses to mix experiments; `adopt` / `import` / `verify` / `list` | src/utils/experiment.py, scripts/experiments.py, run_full_study.py, run_repeated_experiment.py, run_hypothesis_tests.py, run_identity.py, logging_utils.py |
| 2 balance | audit of shared hyper-parameters, loss weight, rows seen, parameters (5%), input dimensions, no per-arm tuning; blocks a launch; compute budget recorded | src/evaluation/balance_audit.py, scripts/check_balance.py |
| 3 replication | paired-by-seed tests (t, Wilcoxon, sign test, 8-of-10 rule, paired TOST); row-level Diebold-Mariano rule from saved forecasts | src/evaluation/paired_tests.py, temporal_validation.py, prediction_dump.py |
| 5 units/tests | analysis plan per hypothesis (5 elements); `planned_vs_ran` in every results file; decidability gates -> verdict NOT_DECIDABLE (H3 poisoning floor, H5 detector at chance) | config/analysis_plan.yaml, src/evaluation/analysis_plan.py, decidability.py, verdicts.py |
| 4 temporal | persistence skill by test block and VIX regime, block bootstrap CI; labelled within-test-period stability, not rolling-origin | src/evaluation/temporal_validation.py, scripts/run_temporal_validation.py |
| 6 preprocessing | five variants (volume winsor, robust scaler, no interpolation, documented features, baseline) in separate folders with their own checkpoints; cheap-proxy sensitivity study and comparison | src/data/variants.py, scripts/prepare_data.py, run_sensitivity_study.py, compare_sensitivity.py |
| deviations | `--record-deviation` logs post-freeze edits of tracked files; margins/alpha/seeds can never be recorded away | scripts/freeze_preregistration.py, PREREGISTRATION_ADDENDUM.md |
| 7 scope | manuscript: background, statement of the problem (4 places), research-question introduction, scope-of-evidence note, Chapter 5 test list, balance audit, analysis-plan table, walk-forward sentence, feature narrative, preprocessing sensitivity, limitations | Word file, tracked changes |

Tests: 395 pass in the sandbox (309 before, 86 new); the 12 failures and 17 collection errors are identical to the untouched
code and come from missing torch / pennylane / data.
