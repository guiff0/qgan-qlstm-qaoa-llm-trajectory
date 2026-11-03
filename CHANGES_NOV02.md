# Nov 02 changes: fair generator comparison + pre-registration

Code (apply on top of the Nov 01 tree):
- `src/utils/generator_path.py`: `make_generator_path_head(..., active_dims=k)`; only the first k standardised features reach G (rest = 0).
- `src/baselines/classical_gan_llm.py`: new `generator_input_dims` (passed as `active_dims`).
- `src/baselines/qgan_llm.py`: head built with `active_dims=n_qubits` (the circuit only encodes those, so behaviour is unchanged but explicit).
- `src/utils/gan_training.py`: `generator_input_dims` is part of the generator-mode fingerprint, dropped in head_only mode.
- `config/default_config.yaml`: `classical_gan_llm.generator_input_dims: 20`; new `classical_ablations` block with `Classical GAN-LLM (matched)` (hidden 6, latent 20, 896 params vs ~912).
- `src/experiments/run_all.py`: expands `classical_ablations` like `ablations`.
- `src/utils/capacity_match.py` (new): parameter-count formula (reproduces 87,072) and `match_hidden`.
- `scripts/run_full_study.py`: `Classical GAN-LLM (matched)` added to PRIMARY conditions (8 conditions).
- `scripts/run_hypothesis_tests.py`: `H4_matched_capacity` reported beside `H4_quantum_advantage`.
- `scripts/freeze_preregistration.py` (new): writes/checks `results/preregistration_<UTC>.json`.
- `tests/test_matched_comparison.py` (new), `tests/test_run_full_study.py` (condition count).

Docs: `PREREGISTRATION.md` (questions, fixed design, decision rules, limitations), `CHAPTER_FRAMING.md` (wording per outcome).

Verified in a sandbox without torch/pennylane: new tests pass (torch ones skip); the rest of the suite fails exactly
where it failed before. NOT verified: the torch paths (`active_dims` gradient test, GAN training in generator mode), and no
real data was available, so no confirmatory result exists yet.

Run order: `python -m scripts.freeze_preregistration` -> `python -m scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 8` (then 12)
-> `python -m scripts.run_full_study` -> `python -m scripts.freeze_preregistration --check`.
