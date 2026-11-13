# Hypotheses and research questions after Amendment 3 (12 Nov 2026)

Five hypotheses (H1-H5), five research questions (RQ1-RQ5), one hypothesis per question. The entanglement sweep and its
hypotheses (old H2, old H5, M2) are dropped to save compute: **64 -> 40 runs** (4 conditions x 10 seeds; the sweep's
4 levels x 8 seeds = 32 runs are gone, and `--full` no longer means 80).

| New | Question | Contrast (expected direction) | Outcome column | Conditions | Came from |
|---|---|---|---|---|---|
| H1 | RQ1 fidelity | QGAN-LLM FID < Classical GAN-LLM FID (ANCOVA on VIX; TOST 10 %) | `fid` | QGAN-LLM, Classical GAN-LLM | old H1 (unchanged) |
| H2 | RQ2 forecast accuracy | QGAN-LLM one-step RMSE < Classical GAN-LLM (TOST 2 % of persistence RMSE) | `rmse` | QGAN-LLM vs Classical GAN-LLM, bracket vs matched | old H4, RMSE half |
| H3 | RQ3 adversarial resilience | QGAN-LLM attack success rate < Classical GAN-LLM (TOST 5 points) | `asr` | same as H2 | old H4, ASR half |
| H4 | RQ4 noise and poisoning | no-noise poisoning degradation > noisy (TOST 5 points; decidability gate + positive control) | `poisoning_rmse_degradation_pct` | QGAN-LLM (no noise) vs QGAN-LLM | old H3 |
| H5 | RQ5 noise and evasion attacks | no-noise ASR > noisy ASR (TOST 5 points) | `asr` | same pair as H4 | new, second half of the old H3 |

## Why H3 was split rather than H1
The old H3 ("noise injection improves resistance") bundled two different attacks. Data poisoning (H4) and evasion attacks
FGSM / PGD / CW (H5) are measured by different columns, through different code paths, and can disagree. `asr` is already scored
for every run, and the no-noise / noisy pair is already in the plan, so H5 costs **no extra runs**. Splitting H1 would have meant
inventing a second fidelity metric (MMD or Wasserstein) that measures the same thing as FID and answers the same question.

## What was not changed
Directions, alpha = 0.05, every equivalence margin, the Holm family (now H1-H5), the advantage rule, the seed rule, and
"inconclusive is never a trend". H2 and H3 are the two metrics that the old H4 already reported separately, so no new test
statistic is introduced. Matched-capacity comparisons are reported beside H2 and H3 as `H2_matched_capacity` and `H3_matched_capacity`.

## Sweep fix
The former sweep level L4 (ring, depth 4, 12 qubits) was the primary QGAN-LLM (12 qubits, ring, 4 layers, noise 0.01) run twice
under two names. It was removed and the levels renumbered L0-L5 (none-d4, linear-d1, linear-d2, linear-d4, full-d2, full-d4).
The sweep is optional now: `--levels 0,2,4` or `--levels all` on the launcher, and `--exploratory-entanglement` on
`scripts.run_hypothesis_tests` (reported as X1-X3, never in the Holm family).

## Rerun rules
* Old checkpoints and result rows do not carry over (names, plan and hypothesis keys changed). Start a new experiment:
  `python -m scripts.run_full_study --new-experiment --base-seed 53` (seeds 53-62, as Amendment 2 set aside).
* Record the change before launching:
  `python -m scripts.freeze_preregistration --record-deviation "Amendment 3: dropped entanglement hypotheses, H4 split into H2/H3, H3 split into H4/H5, duplicate sweep level removed"`.
* The hypothesis labels changed meaning: H2, H3, H4, H5 are not the old H2, H3, H4, H5. Use the table above in Chapter 3 and 4.

## Three-arm steps in the launcher (13 Nov 2026)
`scripts\run_full_study.ps1` now runs `python -m scripts.run_walk_forward` and `python -m scripts.run_naive_baseline --experiment-id legacy`
before the study (skip both with `--skip-three-arm`, refit both with `--redo-three-arm`). They feed the methodology-compliance table (M1-M7,
new M4b) and the three-arm ledger that `scripts.run_hypothesis_tests` prints. They do NOT feed the five verdicts H1-H5, and they do not feed
the Q1 three-arm verdict either: that one reads `results/feature_map_advantage.json` (`scripts.run_feature_map_experiment`).
Fixes: Arm 1 and Arms 2/3 now attack the same rows in every (window, seed) cell; a naive-baseline or walk-forward file fitted with another
ridge grid / plan / attack protocol is refitted instead of reused; the hypothesis tests find the baseline file whatever the experiment id.

## Q1 step and ridge tuning (13 Nov 2026)
* The launcher also runs `python -m scripts.run_feature_map_experiment --seeds 0..9 --n-qubits 8 --skip-if-current` (step 3/3). Q1 needs
  >= 10 paired seeds, else the verdict is UNDERPOWERED. 12 qubits is not run by the launcher (about 20x the simulation cost per row).
* Ridge tuning: half-decade grid (33 values, 1e-10 ... 1e6). An optimum on an end of the grid is followed outward (x100 / /100, up to
  4 steps) for EVERY arm. Each fit records `ridge_status`: interior | saturated_null (the validation set prefers the intercept-only
  forecast = persistence; a finding, not under-tuning) | edge_unresolved (the only status M6 fails on).
* If all three arms choose the null model in >= 80 % of seeds, Q1 reports NOT_DECIDABLE instead of EQUIVALENT_TO_CONTROLS: equivalence
  between three identical persistence forecasts says nothing.
