# 8 Nov 2026 - Amendment 2 patch (apply over the repo root)

New files
- PREREGISTRATION_AMENDMENT_2.md          what changed, what did not, rerun rules
- src/evaluation/poisoning_protocol.py    H3 v2 protocol + positive control + calibration (numpy only)
- src/evaluation/equivalence_power.py     runs needed for equivalence (planning aid)
- src/evaluation/mediation_screen.py      EXPLORATORY discovery/replication mediator screen
- src/llm/threat_calibration.py           H5 calibrated threshold, few-shot block, AUROC, ceiling, positive control
- scripts/calibrate_poisoning.py          choose poisoning.shift_sd on pilot seeds, real-only head
- scripts/run_mediation_screen.py         run the screen on results/all_results.csv
- tests/test_amendment2_protocols.py      15 tests (no torch / pennylane / network)

Edited files
- src/attacks/poisoning.py                + inject_triggered_shift
- src/evaluation/poisoning_resistance.py  v2 branch when config `poisoning.protocol` starts with "v2"; legacy otherwise
- src/evaluation/decidability.py          poisoning_gate(positive_control=), detector_gate(pc_passed=)
- src/llm/llm_pipeline.py                 optional threat_* instrument; legacy columns unchanged when off
- scripts/run_hypothesis_tests.py         pc column into H3 gate; H5 prefers llm_threat_accuracy_cal; H2 power + entropy ladder
- config/default_config.yaml              `poisoning:` section, `llm_pipeline.threat_*` keys (model sections untouched -> no checkpoint fingerprint change)

Order to run
1. python -m scripts.calibrate_poisoning            -> paste shift_sd, set calibrated: true
2. python -m scripts.freeze_preregistration --record-deviation "Amendment 2: H3 positive control, H5 calibrated detector"
3. python -m scripts.run_full_study --new-experiment --base-seed 53 --n-runs 10 --label "Amendment 2"
4. python -m scripts.run_hypothesis_tests
5. (optional, exploratory) python -m scripts.run_mediation_screen
