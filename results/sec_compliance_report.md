# SEC technical-control screen

_Not legal advice or a compliance certification._

## QGAN-LLM (qgan_llm_no_noise_20261025T085629Z): NOT_READY

| ID | Control | Status | Evidence |
|---|---|---|---|
| DOC-1 | Run is reproducible from recorded code, seed and versions | NOT_EVIDENCED | commit abeff9ea recorded but no git_dirty flag (uncommitted edits cannot be ruled out); seed=42 |
| MR-1 | Forecast beats a naive persistence baseline | FAIL | RMSE is 4.513x the one-step persistence RMSE (limit 1.0) |
| CTRL-1 | Adversarial input robustness (FGSM/PGD/CW directional flips) | PASS | ASR=0.2356% over 744785 samples (tolerance 5.0%) |
| CTRL-2 | Threat detection performance | NOT_EVIDENCED | detector=interim_classical_randomforest, accuracy=0.998, FPR=0.001; interim classical detector, identical across models so it does not test this model |
| PRIV-1 | Synthetic data not trivially distinguishable from real (membership test) | FAIL | accuracy=1.000 (chance 0.5); higher = synthetic data trivially distinguishable from real, so it is not a privacy-preserving substitute |
| LAT-1 | Latency vs study-internal target | NOT_EVIDENCED | 0.03233 ms times only a linear head; the quantum generator is not on the inference path, so this says nothing about quantum latency |
| MET-1 | Reported resilience metrics are interpretable | FAIL | QGOM exactly 0.0 (old clamp artifact; below noise floor, not 'no obfuscation'); CTR undefined (no identifiable decay); CTR is a 10-trajectory x 5-step rough estimate; TaLIS stored before the percent/fraction fix; recompute |
| NI-1 | Pre-trade order/credit limits, kill switch, erroneous-order blocks | NOT_IMPLEMENTED | The pipeline forecasts prices; it has no order path or hard limits. Required for any live market access. |
| MAN-1 | Documentation, supervisory procedures, annual review and CEO certification | MANUAL | missing attestations: risk_control_documentation, supervisory_procedures, annual_review_date, ceo_certification |

## QGAN-LLM (qgan_llm_linear_topology_20261024T124334Z): NOT_READY

| ID | Control | Status | Evidence |
|---|---|---|---|
| DOC-1 | Run is reproducible from recorded code, seed and versions | NOT_EVIDENCED | commit abeff9ea recorded but no git_dirty flag (uncommitted edits cannot be ruled out); seed=42 |
| MR-1 | Forecast beats a naive persistence baseline | FAIL | RMSE is 4.513x the one-step persistence RMSE (limit 1.0) |
| CTRL-1 | Adversarial input robustness (FGSM/PGD/CW directional flips) | PASS | ASR=0.2356% over 744785 samples (tolerance 5.0%) |
| CTRL-2 | Threat detection performance | NOT_EVIDENCED | detector=interim_classical_randomforest, accuracy=0.998, FPR=0.001; interim classical detector, identical across models so it does not test this model |
| PRIV-1 | Synthetic data not trivially distinguishable from real (membership test) | FAIL | accuracy=1.000 (chance 0.5); higher = synthetic data trivially distinguishable from real, so it is not a privacy-preserving substitute |
| LAT-1 | Latency vs study-internal target | NOT_EVIDENCED | 0.0215 ms times only a linear head; the quantum generator is not on the inference path, so this says nothing about quantum latency |
| MET-1 | Reported resilience metrics are interpretable | FAIL | QGOM exactly 0.0 (old clamp artifact; below noise floor, not 'no obfuscation'); CTR is a 10-trajectory x 5-step rough estimate; TaLIS stored before the percent/fraction fix; recompute |
| NI-1 | Pre-trade order/credit limits, kill switch, erroneous-order blocks | NOT_IMPLEMENTED | The pipeline forecasts prices; it has no order path or hard limits. Required for any live market access. |
| MAN-1 | Documentation, supervisory procedures, annual review and CEO certification | MANUAL | missing attestations: risk_control_documentation, supervisory_procedures, annual_review_date, ceo_certification |

## QGAN-LLM (qgan_llm_12_qubit_20261024T095145Z): NOT_READY

| ID | Control | Status | Evidence |
|---|---|---|---|
| DOC-1 | Run is reproducible from recorded code, seed and versions | NOT_EVIDENCED | commit abeff9ea recorded but no git_dirty flag (uncommitted edits cannot be ruled out); seed=42 |
| MR-1 | Forecast beats a naive persistence baseline | FAIL | RMSE is 4.513x the one-step persistence RMSE (limit 1.0) |
| CTRL-1 | Adversarial input robustness (FGSM/PGD/CW directional flips) | PASS | ASR=0.2356% over 744785 samples (tolerance 5.0%) |
| CTRL-2 | Threat detection performance | NOT_EVIDENCED | detector=interim_classical_randomforest, accuracy=0.998, FPR=0.001; interim classical detector, identical across models so it does not test this model |
| PRIV-1 | Synthetic data not trivially distinguishable from real (membership test) | FAIL | accuracy=1.000 (chance 0.5); higher = synthetic data trivially distinguishable from real, so it is not a privacy-preserving substitute |
| LAT-1 | Latency vs study-internal target | NOT_EVIDENCED | 0.02513 ms times only a linear head; the quantum generator is not on the inference path, so this says nothing about quantum latency |
| MET-1 | Reported resilience metrics are interpretable | FAIL | QGOM exactly 0.0 (old clamp artifact; below noise floor, not 'no obfuscation'); TaLIS stored before the percent/fraction fix; recompute |
| NI-1 | Pre-trade order/credit limits, kill switch, erroneous-order blocks | NOT_IMPLEMENTED | The pipeline forecasts prices; it has no order path or hard limits. Required for any live market access. |
| MAN-1 | Documentation, supervisory procedures, annual review and CEO certification | MANUAL | missing attestations: risk_control_documentation, supervisory_procedures, annual_review_date, ceo_certification |

## QGAN-LLM (qgan_llm_20261022T060051Z): NOT_READY

| ID | Control | Status | Evidence |
|---|---|---|---|
| DOC-1 | Run is reproducible from recorded code, seed and versions | NOT_EVIDENCED | commit abeff9ea recorded but no git_dirty flag (uncommitted edits cannot be ruled out); seed=42 |
| MR-1 | Forecast beats a naive persistence baseline | FAIL | RMSE is 4.513x the one-step persistence RMSE (limit 1.0) |
| CTRL-1 | Adversarial input robustness (FGSM/PGD/CW directional flips) | PASS | ASR=0.2356% over 744785 samples (tolerance 5.0%) |
| CTRL-2 | Threat detection performance | NOT_EVIDENCED | detector=interim_classical_randomforest, accuracy=0.998, FPR=0.001; interim classical detector, identical across models so it does not test this model |
| PRIV-1 | Synthetic data not trivially distinguishable from real (membership test) | FAIL | accuracy=1.000 (chance 0.5); higher = synthetic data trivially distinguishable from real, so it is not a privacy-preserving substitute |
| LAT-1 | Latency vs study-internal target | NOT_EVIDENCED | 0.03023 ms times only a linear head; the quantum generator is not on the inference path, so this says nothing about quantum latency |
| MET-1 | Reported resilience metrics are interpretable | FAIL | QGOM exactly 0.0 (old clamp artifact; below noise floor, not 'no obfuscation'); CTR undefined (no identifiable decay); CTR is a 10-trajectory x 5-step rough estimate; TaLIS stored before the percent/fraction fix; recompute |
| NI-1 | Pre-trade order/credit limits, kill switch, erroneous-order blocks | NOT_IMPLEMENTED | The pipeline forecasts prices; it has no order path or hard limits. Required for any live market access. |
| MAN-1 | Documentation, supervisory procedures, annual review and CEO certification | MANUAL | missing attestations: risk_control_documentation, supervisory_procedures, annual_review_date, ceo_certification |

## QAOA-Enhanced LLM (qaoa_enhanced_llm_20261024T085701Z): NOT_READY

| ID | Control | Status | Evidence |
|---|---|---|---|
| DOC-1 | Run is reproducible from recorded code, seed and versions | NOT_EVIDENCED | commit abeff9ea recorded but no git_dirty flag (uncommitted edits cannot be ruled out); seed=42 |
| MR-1 | Forecast beats a naive persistence baseline | FAIL | RMSE is 687.1x the one-step persistence RMSE (limit 1.0) |
| CTRL-1 | Adversarial input robustness (FGSM/PGD/CW directional flips) | PASS | ASR=0.0006713% over 744786 samples (tolerance 5.0%) |
| CTRL-2 | Threat detection performance | NOT_EVIDENCED | no detector result |
| PRIV-1 | Synthetic data not trivially distinguishable from real (membership test) | NOT_EVIDENCED | no membership/inversion accuracy logged |
| LAT-1 | Latency vs study-internal target | PASS | 0.1124 ms (unknown) vs STUDY-INTERNAL target 40.0 ms (no SEC numeric threshold exists) |
| MET-1 | Reported resilience metrics are interpretable | FAIL | TaLIS stored before the percent/fraction fix; recompute |
| NI-1 | Pre-trade order/credit limits, kill switch, erroneous-order blocks | NOT_IMPLEMENTED | The pipeline forecasts prices; it has no order path or hard limits. Required for any live market access. |
| MAN-1 | Documentation, supervisory procedures, annual review and CEO certification | MANUAL | missing attestations: risk_control_documentation, supervisory_procedures, annual_review_date, ceo_certification |

## QLSTM Forecaster (qlstm_forecaster_20261023T023234Z): NOT_READY

| ID | Control | Status | Evidence |
|---|---|---|---|
| DOC-1 | Run is reproducible from recorded code, seed and versions | NOT_EVIDENCED | commit abeff9ea recorded but no git_dirty flag (uncommitted edits cannot be ruled out); seed=42 |
| MR-1 | Forecast beats a naive persistence baseline | FAIL | RMSE is 849.5x the one-step persistence RMSE (limit 1.0) |
| CTRL-1 | Adversarial input robustness (FGSM/PGD/CW directional flips) | NOT_EVIDENCED | ASR=2% but only 50 attacked samples (< 1000): one sample moves ASR by 2 points |
| CTRL-2 | Threat detection performance | FAIL | detector=interim_classical_randomforest, accuracy=0.755, FPR=0.0; interim classical detector, identical across models so it does not test this model |
| PRIV-1 | Synthetic data not trivially distinguishable from real (membership test) | NOT_EVIDENCED | no membership/inversion accuracy logged |
| LAT-1 | Latency vs study-internal target | FAIL | 933.6 ms (quantum_circuit_in_path) vs STUDY-INTERNAL target 40.0 ms (no SEC numeric threshold exists) |
| MET-1 | Reported resilience metrics are interpretable | FAIL | QGOM exactly 0.0 (old clamp artifact; below noise floor, not 'no obfuscation'); CTR is a 10-trajectory x 5-step rough estimate; TaLIS stored before the percent/fraction fix; recompute |
| NI-1 | Pre-trade order/credit limits, kill switch, erroneous-order blocks | NOT_IMPLEMENTED | The pipeline forecasts prices; it has no order path or hard limits. Required for any live market access. |
| MAN-1 | Documentation, supervisory procedures, annual review and CEO certification | MANUAL | missing attestations: risk_control_documentation, supervisory_procedures, annual_review_date, ceo_certification |

## Cross-run

- **X-1 FAIL**: 4 GAN-based runs share RMSE=0.000612876785; forecasts come from one closed-form head, so no comparison between them (or quantum effect) is testable

## Manuscript claims

- latency 'mandate' is not in any SEC rule: … as MiFID II and SEC Rule 15c3-5, which require transparency, explainability, and sub-millisecond latency (M. M. Karim et al., 2025). The…
- latency 'mandate' is not in any SEC rule: …oseconds. This directly contradicts the requirement for sub-40ms latency used as a reference value in th…
