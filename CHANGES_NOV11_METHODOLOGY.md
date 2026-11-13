# 11 Nov 2026 - methodology figure, naive-baseline command, H3 poisoning, H5 detector

## 1. `scripts/run_naive_baseline.py` (Arm 1 only)
    python -m scripts.run_naive_baseline --status            # done/pending cells of ALL THREE arms; runs nothing
    python -m scripts.run_naive_baseline --dry-run
    python -m scripts.run_naive_baseline                     # fit Arm 1 on every missing (window, seed) cell, then pair it
    python -m scripts.run_naive_baseline --wait-for-others   # re-check every 5 min until Arms 2 and 3 are complete
    .\scripts\run_naive_baseline.ps1                         # Windows, venv python
Works whether Arms 2/3 are finished or still running: it never opens all_results.csv, logs/study or the manifest, writes
only `results/naive_baseline/<experiment|legacy>/naive_baseline.json` (atomic, resumable), runs at low priority with 1 BLAS
thread, and copies windows/seeds from `results/walk_forward_q8.json` (`--like`) so the cells match the other arms exactly.
Arm 1 is a deterministic ridge, so metrics repeat across seeds inside a window; the windows carry the variance.

## 2. The figure, enforced (`src/evaluation/protocol_compliance.py`)
One check per box (M1 chronological splits, M2 three arms, M3 linear heads, M4 identical five metrics, M5 paired seeds x
windows, M6 equal budget/capacity/tuning, M7 circuit on the inference path, M1b/M8 H3 split + positive control, M9 detector
beats chance). `run_hypothesis_tests` prints the table, stores `methodology_compliance`, and appends a methodology line to the
`why` of EVERY verdict, so no conclusion is stated without it. `FeatureMapForecaster.evaluate` now returns all five metrics
for every arm through `src/evaluation/arm_metrics.py` (same attack rows, same clock). Fidelity there is a forecast-distribution
fidelity (1 - W1/SD), not the generator FID/MMD: do not pool them; record it as a deviation in the pre-registration.

## 3. H3 poisoning strength - was NOT effectively done
`results/poisoning_calibration_20261109T094652Z.json` chose 0.5 with a 662 % median degradation, but the 5 in-run positive
controls degraded 0.20-0.32 % (floor 5 %). Cause: calibration used a sorted random subset split in time; the in-run evaluation
used a random permutation split (which also broke "identical chronological splits, no look-ahead"). Fix: one shared
`poisoning_protocol.chronological_split` (earlier half trains, 1-row embargo, later half scored) used by BOTH; the calibration
grid now starts at 0.05; `--escalate-fractions` raises `fraction` (0.2 -> 0.3 -> 0.4) only if no shift reaches the target;
`apply_calibration` writes shift_sd, fraction and `calibrated: true`. `poisoning.calibrated` is reset to false, so
`run_full_study.ps1` recalibrates first. Rows now carry `poisoning_split`; older rows are flagged FAIL by M1b and must be rerun.

## 4. H5 detector - instrument existed, detector did not beat chance
Held-out AUROC 0.50-0.55 (accuracy 0.50-0.55) in the 5 instrumented rows, while the model-free `last_jump_z` statistic reaches
0.83-0.91 on the same windows. New `llm_pipeline.threat_feature_assist: true` puts that computed statistic in the prompt (and
the few-shot examples). It is reported as "LLM + statistic" (`llm_threat_feature_assist` column), never as the LLM alone.
Prompts changed, so cached LLM answers for the threat task are not reused.

## Must be rerun (not done here: no data, GPU or API access in this sandbox)
1. `python -m scripts.run_walk_forward` (arms 2/3 now record all five metrics), then `python -m scripts.run_naive_baseline`.
2. `.\scripts\run_full_study.ps1 ...` (recalibrates, then H3/H5 rows are regenerated); record the amendment:
   `--record-deviation "Amendment 3: chronological poisoning split, feature-assisted detector, forecast fidelity"`.
Tests: `tests/test_methodology_protocol.py` (10 pass; 11 older tests fail identically on the untouched zip: no torch/Windows here).

## 12 Nov addendum - M6 (edge of the ridge grid)
`DEFAULT_RIDGE_GRID` widened from 1e-8..1e2 (6 values) to 1e-10..1e6 (9 values), shared by all three arms, so no arm can sit on the edge by construction of the old range. Record it as an amendment. Walk-forward results must be rerun.

## 12 Nov addendum 2 - 12 qubits for the whole study
Primary `qgan_llm.n_qubits` 20 -> 12 (also `qlstm_forecaster`, kept identical on purpose); classical generator input dims 12;
matched classical generator re-sized by `capacity_match` to hidden=5, latent=12 (572 vs 560 circuit parameters, 2.1 % apart;
audit passes). The old "QGAN-LLM (12-qubit)" ablation is now the primary, so it became an optional "QGAN-LLM (20-qubit)" width
check (not in the default study). Primary pair and entanglement sweep now share one width. Record as an amendment:
"primary circuit 12 qubits (was 20); matched generator hidden 5". Old 20-qubit checkpoints no longer match (new fingerprints).
Note: sweep level ent-L4 (ring, depth 4) is now configuration-identical to the primary QGAN-LLM.
