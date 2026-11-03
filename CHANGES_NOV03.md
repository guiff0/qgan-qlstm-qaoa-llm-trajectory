# Nov 03 changes (on top of nov02_patch)
- scripts/verify_chapter4.py: removed the hard-coded dissertation claims (DISSERTATION_CLAIMS is empty); prints design checks
  (matched classical comparator present, >= 10 seeds needed); no longer says "within 1% of the dissertation".
- scripts/run_hypothesis_tests.py: QAR and CQRS_per_model no longer written to the results JSON (functions kept, tests unchanged).
- scripts/run_full_study.py: default seeds per condition 5 -> 10 (matches PREREGISTRATION.md); 8 conditions x 10 seeds = 80 runs.
- src/baselines/*_gan_llm.py: per-epoch discriminator accuracy (d_acc) logged.
- src/evaluation/loss_landscape.py, scripts/run_loss_landscape.py, tests/test_loss_landscape.py: new, descriptive only.
- PREREGISTRATION.md: three-arm design, primary metrics, list of retired claims/metrics.
Not run here: anything needing torch/pennylane or the study data; no confirmatory result exists.
