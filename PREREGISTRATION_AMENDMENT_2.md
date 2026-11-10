# Amendment 2 to the pre-registration (8 Nov 2026)

`PREREGISTRATION.md`, `PREREGISTRATION_ADDENDUM.md` and `results/preregistration_*.json` are NOT edited. This amendment is
written after the 39-run sweep was analysed, so it says what was seen and what is allowed to change because of it.

## What the first sweep showed (all numbers from results/hypothesis_test_results.json, 8 Nov)
| Hypothesis | Verdict | Cause that is visible in the data |
|---|---|---|
| H2 entangling structure -> ASR | NOT_SUPPORTED; equivalence not shown (p_TOST = 0.089) | r = -0.085, 95 % CI [-0.39, 0.24], n = 39. The manipulation check passes (rho = 0.48, p = 0.002) and the designed level has no dose-response on ASR (rho = 0.02, p = 0.91). Needs ~57 runs to show \|r\| < 0.3 if r stays at -0.085. |
| H3 noise -> poisoning resistance | NOT_DECIDABLE | The attack reflected 5 % of targets about their mean. Largest degradation in any of 70 logged runs: 0.17 %; median ~1e-5 %. The attack was a no-op in distribution. |
| H5 entanglement -> detection accuracy | NOT_DECIDABLE | Zero-shot LLM pooled accuracy 0.501 over 9,168 windows; FPR ~ recall ~ 0.76 (says "threat" for everything). Between-run SD 0.007 < binomial noise 0.035. |
| M2 mediation by entropy | NOT_SUPPORTED | Indirect -0.069, CI [-0.207, 0.037]. There is no total effect (c = 0.128, n.s.) to mediate. Entropy by level is NOT monotone (L0 0, L2 0.15, L4 0.41, L6 0.10), so a linear level -> entropy path (a = 0.41) is misspecified. |

## What changes (instrument fixes, none of which looks at a hypothesis contrast)
1. **H3 attack and positive control** (`src/attacks/poisoning.py::inject_triggered_shift`, `src/evaluation/poisoning_protocol.py`, config `poisoning:`).
   Feature-triggered target shift, closed-form ridge head, and a positive control (the same attack on a real-rows-only head,
   column `poisoning_pc_degradation_pct`). `poisoning_gate` makes H3 NOT_DECIDABLE unless the control's median degradation
   >= 5 points (the existing H3 margin). Strength is calibrated once by `scripts/calibrate_poisoning.py` on pilot seeds
   100-104 with the real-only head: no generator, no noise level, no confirmatory seed is involved. Value frozen in config.
2. **H5 detector** (`src/llm/threat_calibration.py`, `src/llm/llm_pipeline.py`, config `llm_pipeline.threat_*`).
   Threshold calibrated on 30 % of windows and accuracy scored on disjoint evaluation windows (`llm_threat_accuracy_cal`);
   4 labelled pairs in the prompt (never scored); AUROC; a detectability ceiling (`ref_last_jump_auroc`); and a positive
   control on a 20x-SD perturbation (`llm_threat_pc_passed`). `detector_gate` makes H5 NOT_DECIDABLE if the detector misses
   the gross perturbation in more than 20 % of runs.
3. **H2 power** is reported (`equivalence_r.runs_needed_if_r_unchanged`) and the entropy ladder is reported per level
   (`entropy_by_level`, `entropy_ladder_monotone`). Descriptive only.

## What does NOT change
Directions of every hypothesis, alpha = 0.05, equivalence margins (r0 = 0.30, ASR 5 pts, poisoning 5 pts, RMSE 2 %,
FID 10 %), the Holm family, the advantage rule, the seed rule, "inconclusive is never a trend".

## What was deliberately NOT done
* No alternative H2 relationship and no alternative M2 mediator was adopted. Choosing a variable because it correlates with
  ASR on the runs already analysed is a new hypothesis tested on its own discovery data. `src/evaluation/mediation_screen.py`
  exists for hypothesis generation only (discovery half of the seeds, Holm over candidates, replication half, label
  EXPLORATORY). On the 39 runs: 14 candidates, 0 survivors (`results/mediation_screen_*.json`). `eer` excluded 0 in the
  discovery half alone, which is what ~0.7 expected false hits looks like.
* H2/M2 are not re-specified. If the rerun gives \|r\| < 0.3 with a CI inside the margin, that is the finding: "no practically
  relevant relationship in this design".

## Rerun rules (declared before running)
* New experiment id (`--new-experiment`), fresh seeds (43-52 are used; use **53-62**), 10 seeds per condition, no optional stopping:
  the number of seeds is fixed now and is not extended until a test turns significant or equivalent.
* Re-evaluating a checkpoint is not a new run: re-scoring seeds 43-52 under the new protocol adds no independent H2 data. H2/M2
  may be pooled across experiments only through the existing pooled-experiment path, with the seed-overlap check on.
* H3 and H5 use the new protocol only (old runs have no positive control), so they are not pooled with the old experiment.
* Calibration seeds (100-104) never appear in the confirmatory list; `scripts/calibrate_poisoning.py` refuses overlap.
* Record the change: `python -m scripts.freeze_preregistration --record-deviation "Amendment 2: H3 positive control, H5 calibrated detector"`.

## Not verified in the environment where this was written
No torch, no PennyLane and no NVIDIA key were available. The numpy-only parts (protocol, gates, calibration helpers, screen,
LLM-pipeline logic against a stub LLM) are covered by `tests/test_amendment2_protocols.py` (15 tests). The poisoning `shift_sd`
in the config (3.0) comes from a simulation, not your data: run `scripts.calibrate_poisoning` and replace it. Whether the real
LLM passes the positive control is unknown until it is run.
