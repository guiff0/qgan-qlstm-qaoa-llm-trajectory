# Pre-registration v2 (DRAFT: not frozen until the open-decision marker below is replaced and `scripts/check_prereg_v2.py` passes)

Supersedes the *design* of `PREREGISTRATION.md` for a NEW experiment. Experiment 1 (all 86 rows, `experiment_id = unassigned`, results
file `hypothesis_test_results__20261109T090747Z.json`) stays archived and unedited; this document does not re-score it.

> **STATUS (P0 RESULT, recorded before any v2 data).** `scripts.report_trained_weight_scale` on the existing qgan_llm checkpoints:
> trained circuit-weight SD 0.047-0.117 (median **0.072**), max |theta| 0.41, i.e. essentially unchanged from the 0.1 initialisation.
> Frozen rule: median SD < 0.5 -> **Stage 1 and the dose study (sections 3-5) are NOT run.** Consequences: (i) H2/M2 keep their
> Experiment 1 verdicts (NOT_SUPPORTED) and are reported with the reason "no functioning entanglement dose in the trained
> generators"; (ii) sections 6-8 (H3, H5, three-arm/H1/H4) proceed independently of the dose; (iii) sections 3-5 stay in force
> only if a generator redesign (e.g. larger circuit-weight scale) is first pre-registered as a new amendment and passes P0.

## 0. What Experiment 1 showed (nothing below is re-opened)
| Item | Experiment 1 verdict | Status for the write-up |
|---|---|---|
| H1 FID | CONTRADICTED (4 QGAN vs 10 classical runs; paired rule not met) | reported, underpowered, re-run under section 8 |
| H2 entanglement -> ASR | NOT_SUPPORTED, r = -0.087, n = 40, equivalence not shown (p = 0.088) | valid null; entropy ladder non-monotone (0, 0.15, 0.41, 0.11) |
| H3 poisoning | NOT_DECIDABLE (max change 0.00036 pts) | instrument defect (attack was a no-op), fixed in section 6 |
| H4 quantum advantage | CONTRADICTED (3 vs 9 runs; too few shared seeds) | underpowered; **no configuration beats persistence** (section 8) |
| H5 detector | NOT_DECIDABLE (accuracy 0.500 over 9,780 windows) | instrument defect, fixed in section 7 |
| M2 mediation | NOT_SUPPORTED (indirect CI [-0.20, 0.04]; no total effect) | valid null |

## 1. Unchanged from v1 (cannot be edited after the freeze)
alpha = 0.05; Holm family and advantage rule as in `config/analysis_plan.yaml`; equivalence margins (r0 = 0.30, ASR 5 pts, poisoning 5 pts, RMSE 2 % of persistence, FID 10 %);
"inconclusive is never reported as a trend"; one run = one seed trained from scratch; re-scoring a checkpoint is not a new run; experiment ids; archive-before-adopt.

## 2. Seeds and sample sizes (fixed now, no optional stopping)
- Confirmatory seeds: **53-72** (fresh; 43-52 are spent). Pilot seeds: **100-104** (never in the confirmatory list).
- Dose study: 5 doses x **20** seeds = **100** runs. The positive-control arm (section 3) runs on the pilot seeds as a gate; it is not an extra confirmatory arm.
- Power of the trend test (simulation: linear decline across the doses, between-seed SD 17, one-sided .05): a 15-point total drop -> 0.90 at 20 seeds/dose; 10-point -> 0.63; 20-point -> 0.99.
- Primary H1/H4 conditions (QGAN, classical, classical matched): **10** seeds each, **the same seeds in every condition** (Experiment 1 had 3-4 vs 8-10, which is why the paired rule could not be evaluated).
- H3 conditions (noise vs no-noise): **10** seeds each, same seeds.

## 3. Dose manipulation and Stage 1 gates (pilot seeds only, entropy only)
- Circuit: `rzz:<theta>` replaces the CNOT ring; theta fixed per condition, never trained; theta = 0 emits no two-qubit gate; **last-layer RZZ skipped** (a Z read-out cannot see it). Wired in `src/quantum/entangler.py` and verified by `tests/test_rzz_topology.py`.
- theta grid: **{0, pi/8, pi/4, 3pi/8, pi/2}** (Option C: tunable angle).
- Evidence for the grid/scale decision (`scripts.simulate_entropy_dose`, 12 qubits, 4 layers, random weights): the dose is monotone in the MEAN only when the rotations are large. Weight SD 1.0 / 2.0 / 3.0: mean curve increasing, 87 / 97 / 97 % of single draws strictly increasing. Weight SD 0.1 and 0.3: mean curve NOT increasing (rank rho +0.60 / +0.90). A lower grid (0..pi/4) and a depth-based dose (1-6 layers) behave the same way: **the weight scale, not the grid, decides**.
- **P0 screen before any pilot run:** `python -m scripts.report_trained_weight_scale --simulate`. If the median trained weight SD is below 0.5, Stage 1 is expected to fail and is not run; the generator's initialisation is redesigned and this document amended first. Weight scale is a property of the trained generators, so it is measured, not assumed.
- Entropy reported = bipartite von Neumann entropy (half register) of the circuit state at the fixed all-zeros reference input with the trained weights, i.e. the pre-read-out state (`QGANLLM._measure_entanglement`). It is a property of circuit + weights at one input, not an average over data.
  **Measurement caveat (found at P0):** at the trained weight scale (SD ~0.07) an RZZ dose measured at x = 0 gives S_E of ~0.003 bits at every theta (RZZ is diagonal; on |0...0> it adds only a phase), while data-like inputs (x ~ N(0,1)) give 0.0-0.5 bits and a non-monotone curve. If a redesign revives this section, S_E must be averaged over a fixed set of data inputs (new column `entanglement_entropy_data`, implemented as the mean over 8 fixed N(0,1) inputs drawn with seed 7919, independent of the run seed; `purity_data` and `entanglement_entropy_data_sd` are recorded alongside; the original x = 0 column is unchanged), not taken at x = 0; the Experiment 1 CNOT-topology entropies are not comparable to RZZ values.
- G1 (all three required): Jonckheere-Terpstra increasing p < **0.05**; mean entropy curve strictly increasing; Spearman rho >= **0.7** with one-sided p < alpha. Per-seed strict ordering is informational only.
- G2 positive control: `QGAN-LLM (dose-PC untrained)` = theta-0 architecture with `untrained: true` (zero GAN epochs); |ASR gap vs the trained theta-0 arm| >= **10** points in >= **4** of 5 pilot seeds. (A 5-point per-seed gap is not resolvable against a between-seed SD of ~17.)
- If P0, G1 or G2 fails: the confirmatory dose study is NOT run; H2 is re-specified (e.g. as a contrast) in a new amendment before any data.

## 4. H2 (revised; CONDITIONAL on a redesign that passes P0, see STATUS)
- H2a (primary): Jonckheere-Terpstra, ASR decreasing across the 5 doses, one-sided alpha = .05. Minimum effect of interest: a **15**-point total drop (power 0.90 at 20 seeds/dose). A 5-point effect is NOT detectable here: seed-to-seed SD ~17 gives a standard error of the dose mean of ~3.8 points at 20 seeds.
- H2b (kept from v1, secondary): run-level correlation equivalence, |r| < 0.30 (90 % Fisher-z CI). With 100 runs it can be shown when the observed |r| is up to about 0.14; above that it fails. An ASR-difference TOST at +/-5 points is NOT used: it would need ~200 seeds per group.
- Verdicts: SUPPORTED / CONTRADICTED (by sign) / EQUIVALENT (H2a not rejected and H2b shown) / INCONCLUSIVE / NOT_DECIDABLE (positive control G2-type gate failed).
- ASR definition unchanged from Experiment 1: `adversarial.attacks` (fgsm, pgd, cw) pooled as configured. Raising the number of attacked samples is a stability diagnostic, not a power lever (between-seed variance dominates), and changing the attack set would change the dependent variable, so neither is done.

## 5. M2 (revised; CONDITIONAL on a redesign that passes P0, see STATUS)
- Preconditions: significant total effect (H2a rejects), monotone dose -> entropy path (G1 passed), >= 30 runs. Otherwise **NOT_DECIDABLE**, not null.
- Test: Baron-Kenny, bootstrap 5,000 resamples, percentile 95 % CI; mediator = measured entropy only; SUPPORTED needs the indirect CI to exclude 0 and |c'| < |c|. Purity and QGOM_d are secondary, reported with Holm, and cannot support M2 (purity is nearly collinear with entropy; QGOM_d has no derived direction).
- Wording if SUPPORTED: "consistent with mediation through measured entropy". The b-path rests on non-randomised seed-to-seed variation; no-unmeasured-confounding is untested; "entanglement causes" is not claimed.

## 6. H3 (revised)
- Protocol v2: feature-triggered target shift, fraction **0.2**, shift_sd = [FILL: value printed by `python -m scripts.calibrate_poisoning` on pilot seeds, then frozen with `calibrated: true`], closed-form ridge head.
- Positive control (real-rows-only head, same attack): median degradation >= **5** points, else H3 is NOT_DECIDABLE. Calibration never sees a generator, a noise level or a confirmatory seed.
- Test: Welch + TOST (5 pts) no-noise vs noisy on matched configs, as in v1. The outcome in the hypothesis is the DIFFERENCE between noisy and no-noise degradation (negative = noise protects); degradation itself is positive for a working attack.

## 7. H5 (revised)
- Detector instrument (config `llm_pipeline.threat_*`): threshold calibrated on **30 %** of the windows and scored on the disjoint rest; **4** few-shot pairs (never scored); positive control = gross perturbation **20 x** the 1-minute SD, detector accuracy >= **0.75** in >= **80 %** of runs.
- H5 gate: pooled calibrated accuracy above chance (one-sided exact binomial, alpha .05) AND the positive control passed; else NOT_DECIDABLE. Diagnostics reported, not gating: AUROC and the last-jump detectability ceiling.
- Test: Pearson r between measured entropy and detector accuracy (Fisher-z 95 % CI) with the equivalence companion |r| < 0.30, same structure as H2.
- Windows per run as configured (`llm_pipeline.n_windows`); the earlier "9,168 windows" is the pooled count across Experiment 1 runs, not a per-run size.

## 8. Quantum advantage (H1, H4, three-arm)
- **Persistence gate (frozen before Stage 2):** a quantum-vs-classical comparison is read as a forecasting result only if at least one arm beats persistence on the shared test rows (RMSE/persistence below 1 with the whole 95 % CI below 1). Otherwise the comparison is reported as **NO_FORECASTING_SIGNAL** and H1/H4 are not read as advantage. The same gate applies to the three-arm experiment.
- Arms (as in code): inputs only, classical random features, quantum circuit features; all forecast from [x, phi(x)] (nested, the stricter control). The diagram showing phi(x) alone is updated to match the code.
- Matched resources (balance audit must PASS): parameter budget (classical random features capped at n_layers x n_qubits x 3 of the circuit; fixed at 12 qubits), feature dimension, ridge grid, rows, seeds.
- Rule (unchanged from v1 Q1): ADVANTAGE_SUPPORTED needs the quantum arm to beat BOTH controls (paired Diebold-Mariano p < .05 in >= 8 of 10 seeds) AND skill > 0; else EQUIVALENT (paired TOST +/- 2 % of persistence RMSE) / CONTRADICTED / INCONCLUSIVE / UNDERPOWERED (< 10 seeds).
- Sensitivity matrix (`scripts.run_three_arm_matrix`): matched cells (training rows full/50 %/25 %, qubit width, ridge-grid size) and bracket cells (classical width x2/x4, 4x finer classical ridge grid). An advantage that disappears in the bracket cells is not credited to the circuit.
- Synthetic-sample count and loss weights do not exist in the feature-map study (no generator; one unweighted loss) and are reported as not applicable; the generator study's are audited by `balance_audit`.
- Forecast test rows and the persistence RMSE are computed on the same rows, with the subsample size as configured.

## 9. Variables (operational definitions)
| Variable | Role | Definition | Direction |
|---|---|---|---|
| theta | IV (design) | shared RZZ angle on ring edges; fixed per condition; 5 values | none |
| S_E | mediator (M2) | half-register von Neumann entropy, trained weights, all-zeros reference input, exact statevector | rises with theta (gate G1) |
| purity | secondary | Tr(rho^2), same reduced state | falls with theta; collinear with S_E |
| QGOM_d | secondary | 1 - Var(g_attacked)/Var(g_clean), finite-shot parameter-shift gradients (`src/quantum/gradient_obfuscation.py`) | none pre-specified |
| ASR | DV (H2) | attack success rate pooled over the configured attacks | H2a: decreases with theta |
| poisoning degradation | DV (H3) | % RMSE increase of the poisoned vs clean head | H3: smaller with noise |
| positive-control gap | gate | ASR(untrained theta-0 generator) - ASR(trained theta-0 generator) | magnitude >= 10 points |
| seed | replication unit | one training run from scratch | none |

## 10. Exploratory (labelled, never verdicts)
Mediator screen (`scripts.run_mediation_screen`), entropy-bin analysis (quartiles, Kruskal-Wallis), Kendall tau and Page's L as secondary trend tests, sensitivity of entropy to the estimator, QGOM_d and purity as secondary mediators, the bracket cells above. Anything found is a hypothesis for a new pre-registered run.

## 11. Rules
No change to doses, thresholds, seeds or margins after Stage 1 results are seen. Pooling across experiments only via the existing pooled-experiment path with seed-overlap check; v2 H3/H5 runs are not pooled with Experiment 1.
Unverified figures (hours per run, manuscript table values such as 1.379 vs 1.077) are NOT cited until checked against `results/all_results.csv` and the logs.

## 12. Pre-Stage-1 checks and freeze procedure
1. `python -m scripts.report_trained_weight_scale --simulate` (P0 screen).
2. Untrained-mode smoke test: add the six dose conditions (`python -m scripts.calibrate_entanglement_dose --print-config`) to the config, then run ONE seed of the control:
   `python -m scripts.run_full_study --new-experiment --base-seed 100 --n-runs 1 --conditions "QGAN-LLM (dose-PC untrained)" --label "untrained-mode smoke test" --allow-uncalibrated`.
   Verify in the log: no GAN epochs, "Generator-path head fitted on [X, G(X)]", and `untrained_control = True` in the run record. If the flag is absent the positive-control gate cannot fire and H2/H3 are NOT_DECIDABLE.
3. `python -m scripts.calibrate_poisoning` on the pilot seeds; set shift_sd and `calibrated: true`.
4. `python -m scripts.check_prereg_v2` must pass (no open markers).
5. `python -m scripts.freeze_preregistration`, then `--record-deviation "Amendment 2: H3 positive control, H5 calibrated detector, H2 dose design, persistence gate"`.
6. Stage 1 (pilot seeds 100-104): G1 and G2 via `python -m scripts.calibrate_entanglement_dose`; then Stage 2 via `python -m scripts.run_study`.

## 13. Code-fix deviations (recorded after Experiment 1; nothing here re-scores Experiment 1)
- **D1, n per group for the confirmatory tier (bug fix against the frozen plan).** `src/evaluation/verdicts.py` used `MIN_N_PER_GROUP = 3` to assign the CONFIRMATORY tier of H1, H3 and H4, while `config/analysis_plan.yaml` requires `minimum_for_confirmatory: 10` for each. The code now reads a per-hypothesis table (`MIN_N_CONFIRMATORY_PER_GROUP`) that `tests/test_h4_verdict_logic.py` keeps equal to the plan. A comparison with fewer than 10 runs in either arm can no longer return SUPPORTED or CONTRADICTED; a significant result is reported as SUGGESTIVE_FOR / SUGGESTIVE_AGAINST (exploratory).
- **Experiment 1 is not re-scored.** Its H4 row stays as computed (CONTRADICTED, 3 vs 9 runs) and is reported as exploratory because the plan's confirmatory minimum was not met; the write-up discloses that the code did not enforce that minimum. The corrected logic is validated on synthetic cases only.
- **Known behaviour, deliberately not changed.** `verdict_h4` ranks CONTRADICTED, SUPPORTED, INCONCLUSIVE, NOT_SUPPORTED and does not rank SUGGESTIVE_*, so with one significant exploratory metric and one non-significant metric the headline is INCONCLUSIVE. It is pinned in the tests; changing it is a new rule and needs its own amendment before new-study data.
- **Not adopted here (design changes, each needs its own amendment before new-study data):** paired-by-seed as the verdict-forming H4 statistic, splitting H4 into an RMSE verdict and a secondary ASR line, and a forecast-agreement degeneracy flag (new code; it can only run on runs whose per-row predictions were saved).

