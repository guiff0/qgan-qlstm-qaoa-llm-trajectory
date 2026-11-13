# Amendment 3 (13 Nov 2026) - recorded before any confirmatory result of the new design exists

Hypotheses and plan: see `HYPOTHESES_V3.md` (five hypotheses H1-H5, entanglement sweep dropped, 4 conditions x 10 seeds = 40 runs).

## Independence of the three-arm study (stated, enforced, tested)
`run_walk_forward`, `run_naive_baseline` and `run_feature_map_experiment` have no generator, no LLM, no QAOA and no torch/PennyLane on
their path (`tests/test_three_arm_consistency.py::test_the_three_scripts_import_no_generator_llm_qaoa_or_torch_even_when_they_run`).
Consequences: (1) their compliance (M1-M7, M4b) is attached to the Q1 result only and is no longer stamped on H1-H5; the generator-side
checks M8 and M9 are attached to H4 and X2 only; (2) the files are shared by every experiment (`results/naive_baseline/legacy`).

## Changes to the three-arm measurement protocol (all arms identically)
1. Attack rows / latency rows: every arm of a (window, seed) cell uses `metric_seed = seed` (Arms 2 and 3 used 0 before, Arm 1 used the seed).
2. Ridge tuning: half-decade grid 1e-10 ... 1e6 (33 values, was 9). An optimum on an end of the grid is followed outward (x100 / /100,
   at most 4 steps) while validation MSE improves; same rule for every arm, validation rows only. Each fit records `ridge_status`
   (interior | saturated_null | edge_unresolved). M6 fails only on edge_unresolved. Reason: the old M6 read "the validation set prefers
   the intercept-only forecast" as under-tuning.
3. Q1 guard: if every arm chose the null model in >= 80 % of seeds, EQUIVALENT_TO_CONTROLS becomes NOT_DECIDABLE (all three forecasts are
   the same persistence forecast). The guard can only withdraw an equivalence claim; it cannot create an advantage or change any margin.
4. New check M4b (same test rows, attack rows, attack strength and metric seed in every cell). Stale files (other ridge grid, windows,
   attack protocol, or rows without metric seed) are refitted, never reused.

Unchanged: margins, alpha, the advantage rule (beats BOTH controls, paired DM p < 0.05, same sign in >= 8 of 10 seeds, skill > 0),
the TOST equivalence rule, >= 10 seeds, the seeds, the windows, the splits.

## Disclosure
Items 2-3 were prompted by an M6 failure seen on synthetic data and on earlier, non-confirmatory real runs. They are fixed here, before the
new design's confirmatory runs. `freeze_preregistration` now also hashes the three-arm decision and measurement files, so a later change to
them shows up in `--check`. Run `freeze_preregistration` AFTER the last code change and BEFORE `run_full_study`.

## Q1 widths
`PREREGISTRATION.md` names 8 and 12 qubits. The launcher runs 8 by default and 12 with `--with-12q`; leaving 12 out is a deviation to record.
