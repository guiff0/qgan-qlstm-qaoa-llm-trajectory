# One entry point: `python -m scripts.run_study`

Apply order: (1) your CHANGES_NOV07 zip, phase A (and B if you want), (2) amendment2_patch.zip. They share no files.

```
python -m scripts.run_study --plan                       # every stage with its exact command, nothing executed
python -m scripts.run_study --new-experiment --label "Amendment 2" --n-runs 10 --base-seed 53 --workers 4 \
       --apply-calibration --record-deviation "Amendment 2: H3 positive control, H5 calibrated detector"
python -m scripts.run_study                              # re-launch/resume: finished stages skipped, main stage resumes
python -m scripts.run_study --from hypothesis            # redo the analysis only
python -m scripts.run_study --status
```

Order: calibrate -> freeze check -> main runs -> three-arm feature map (8 and 12 qubits) -> matched/bracket matrix ->
hypothesis tests (experiment id + naive baseline + three-arm files) -> temporal validation (optional) -> mediation screen (optional,
exploratory). A failed required stage stops the study and says why; state is in results/study_driver_state.json.

It stops, on purpose, when: the poisoning strength is not calibrated and you did not pass --apply-calibration or
--allow-uncalibrated; no shift reaches the calibration target (H3 then is not decidable); tracked files drifted from the freeze and
no --record-deviation reason was given; the balance audit inside run_full_study fails.

Not part of it (run by hand once the circuit wiring passes tests/test_pqc_runner.py under PennyLane):
`python -m scripts.calibrate_entanglement_dose --print-config`, then the pilot runs and the same script without the flag.
