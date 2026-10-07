# Codebase TODO — manuscript-vs-code remediation

Prioritized per the 09/28-10/05 conversation. Check off as each lands;
keep this file in sync with what's actually in the repo, not what's
planned.

**7 items waiting implementation** (everything under "Backlog" +
"Recommended, not yet started" below). "Deliberately not code work" is
a separate list — those are manuscript-text fixes, not implementation
gaps, so they don't count toward that number.

## Backlog — not started

- [ ] **Startup prompt for quantum training scale** — `run_pipeline.py`
      should ask the user, once per fresh run: which qubit
      ablation(s) to include (20q ring baseline / 12-qubit /
      linear-topology / no-noise / QLSTM Forecaster), a training-scale
      preset (rows subsampled + epochs — the full 3.7M-row/58k-batch
      schedule is not tractable for a 20-qubit lightning.qubit circuit
      on a single CPU machine), and whether to apply the no-grad fix
      below. Persist the choice in the pipeline's state file so a
      resumed run doesn't re-prompt (matches the existing
      `decide_skip_policy` UX). **Still the single highest-value
      remaining item** — it's what actually makes the repeated-runs
      harness tractable to run at the scale H1-H5 need.
- [ ] `run_all.py`: add `--override key=value` (repeatable) CLI flag,
      applied to the selected model's config after ablation-merging —
      needed by the startup prompt above. (A narrower `--seed` override
      was added for the repeated-runs harness; the general-purpose flag
      is still open.)
- [ ] `run_all.py`: support `max_train_rows` / `max_val_rows` /
      `max_test_rows` per-model config keys — truncate to a
      chronological PREFIX (no shuffling).
- [ ] **No-grad fix in `qgan_llm.py`** — `_train_discriminator_step`
      calls `self.generator(z)` without `torch.no_grad()`. Removes
      autograd/adjoint bookkeeping overhead only, not the dominant
      simulation cost — don't oversell this as fixing the training-scale
      problem.

## Recommended, not yet started

- [ ] **Extend or disclose the cross-validation gap** — ForexSB/HistData
      cross-check is capped at 2023. 2024-2026 have zero independent
      corroboration of the Dukascopy series.
- [ ] **Macro-feature staleness indicator** — a "days since last update"
      feature per macro series in `prepare_data.py`.
- [ ] **Reconsider outlier filtering (MAD 5.0/3.0) against stress
      periods** — check whether known stress windows (2020 COVID, 2022
      rate-hike volatility) got filtered out of a study whose hypotheses
      are specifically about robustness under abnormal conditions.
- [ ] **QCBM-based quantum discriminator** — the manuscript's own
      definition of QGANs needs a quantum discriminator (Quantum Circuit
      Born Machine), not the current classical one. Real architecture
      change, bigger lift than the others.
- [ ] **Walk-forward / expanding-window CV**, replacing the single
      chronological split.
- [ ] **M3->DV3 and M4->DV4 mediation** — `mediation_pipelines.py` has
      both as library functions; `run_hypothesis_tests.py` only calls
      M2 so far. M3 needs multiple independent runs to pull
      `fpr_per_run` from (gated on the repeated-runs item below actually
      being run at scale); M4 needs a `computational_overhead_ms` column
      that doesn't exist in `results/all_results.csv` yet.

## Known limits of what's landed (read before relying on these)

- **LLM Forecaster is zero-shot, not fine-tuned, and UNVERIFIED against
  a live endpoint.** `src/llm/llm_inference.py` and
  `src/baselines/llm_forecaster.py` are real, tested-against-mocks
  integration code, registered in `run_all.py` only when
  `NVIDIA_API_KEY` is set. This development session has no key and no
  network route to api.nvidia.com at all, so the live call has never
  succeeded against a real account — verify the request/response shape
  against NVIDIA's current API before trusting any number it produces.
  Fine-tuning itself (`nvidia_finetune.submit_finetune_job` +
  `poll_finetune_status`) is a separate, hours-long, asynchronous step
  this model's `train()` does NOT run automatically.
- **The repeated-runs harness (`scripts/run_repeated_experiment.py`) is
  real, resumable infrastructure — it does not make 100-200 runs per
  condition fast.** Tested for its own orchestration logic (state
  tracking, resumability) with a mocked subprocess call, since this
  sandbox has no `data/processed/*.npy` to actually train against.
  Running it at manuscript scale is still gated on the quantum-
  training-scale item above.
- **CTR now computes at ANY qubit count, including the primary 20-qubit
  configuration — but via two different methods with very different
  precision.** At ≤12 qubits: exact (dense density matrix). Above that:
  a trajectory (Monte Carlo wavefunction) method
  (`src/quantum/decoherence.py`) that never builds a dense matrix at
  all (O(2^n) memory, not O(4^n) — 16 MB vs 16 TB at n=20). Cross-
  validated against the exact method at a small qubit count (within
  ~0.005 after fixing a real bug — see below) AND against the exact
  analytical single-qubit amplitude-damping decay curve. **But the
  version wired into `resilience_suite.py`'s automatic per-model path
  uses n_trajectories=10, n_steps=5 — a FAST, ROUGH estimate, not a
  publication-quality number** (full cross-validation used
  n_trajectories=1000, which costs ~15-25 minutes per model at 20
  qubits — too slow for routine evaluation). For a real CTR result,
  call `decoherence_rho_series_trajectory` directly with far more
  trajectories as a separate, one-off computation.
  **Two real bugs were found and fixed while validating this:**
  (1) `_depolarizing_kraus` originally used a different, non-equivalent
  parameterization than `qml.DepolarizingChannel` (the one the exact
  method actually calls) — caught because the clean condition (no
  depolarizing channel) matched the exact method perfectly while the
  attacked condition (which uses it) had an error that did NOT shrink
  with more trajectories, the signature of a formula bug, not noise.
  Fixed to match PennyLane's own Kraus matrices exactly, with a
  permanent regression test. (2) `ctr()`'s own `n_qubits` argument is a
  tractability check on the density matrices it's actually handed, not
  a record of the original circuit's size — `resilience_suite.py` was
  passing the full n_qubits (e.g. 14) instead of the REDUCED subsystem
  size the trajectory method actually returns (e.g. 7), causing `ctr()`
  to wrongly reject a perfectly tractable reduced matrix as if a dense
  14-qubit one had been built. Fixed by passing the actual subsystem
  size.
- **Same-row leakage fix changes previously-reported RMSE/FID numbers.**
  Any `results/all_results.csv` row generated before the
  `one_step_ahead.py` fix used the OLD, leaked same-row pairing for
  Classical GAN-LLM, QGAN-LLM, and QLSTM Forecaster — not comparable to
  anything generated after. Classical LSTM was never affected.
- **The 2012-2026 study window now self-extends with zero code
  changes, including gracefully SKIPPING an incomplete trailing year
  instead of aborting the whole pipeline.** `data.test_end` names
  2026-12-31 (15 years), but `_effective_end_year()` caps what's
  actually fetched at the last COMPLETE year — today that's 2025 (14
  years), printed as a clear note, not an error. `acquire_dukascopy()`/
  `acquire_fred()`/`consolidate_dukascopy()` all use this capped value.
  The master file keeps its full "2012_2026" name (the intended final
  state) and grows into it automatically: re-run any time after 2026
  actually closes and 2026 is included with no further action. (The
  OLD behavior — hard `SystemExit` when the configured end year wasn't
  complete — is still available as `_refuse_if_incomplete_year()` for
  anyone who explicitly wants a hard failure instead, e.g. a scheduled
  job that should treat "not done yet" as an error; it's just no longer
  what `acquire_dukascopy()`/`acquire_fred()` call by default.)

**Cross-checked again against the manuscript's actual H1-H5 text:** all
five hypotheses' dependent variables are now produced by the pipeline on
genuinely non-leaked data. The remaining blocker for an actually valid
conclusion is the repeated-runs gap — the harness now exists, but
hasn't been run at scale because the training-scale bottleneck that
would make that tractable is still the top Backlog item.

## Deliberately NOT code work — manuscript text fixes instead

- Data Sanitization / differential privacy — out of scope; note
  regex-only PII scanning as a limitation.
- GenAI "multi-modal" claim — scope down to tabular-only in the text.
- RMSE + decision-trees + k-fold conflation in Ch.1's definition.
- Quantum-Classical Coherence vs. the code's separate "Coherence Time
  Retention" metric — clarifying wording, not new code.
- Single-broker/no-order-book-volume limitation — one limitations-
  section line.
- Automated Vulnerability Assessment, Cyber Threat Intelligence, AI
  Guardrails, Adaptive Resilience/ADT — confirmed background/lit-review
  terms only, never claimed as implemented methodology or results.
- FPR mislabeled as "DV3" in an old code comment vs. Table 19's DV5 —
  a manuscript-internal numbering inconsistency, not a code bug.

## Already resolved this conversation

- [x] `src/quantum/qaoa.py`, `src/quantum/qpca.py`.
- [x] **TaLIS, real implementation** — `src/evaluation/talis.py`.
- [x] **Mode-collapse diagnostics + FID/MMD/Wasserstein now actually
      reach `results/all_results.csv`**.
- [x] **Dukascopy 2012 ingestion bug** (deterministic, not flaky).
- [x] **Extended study window to 15 years (2012-2026)**, now with
      graceful year-skipping (see "Known limits" above) rather than a
      hard abort.
- [x] **H3 wired end-to-end** (poisoning + model inversion) —
      `src/evaluation/poisoning_resistance.py`.
- [x] **H5 wired end-to-end** (threat detection / FPR) —
      `src/evaluation/threat_detection.py` (interim classical detector).
- [x] **Quantum resilience suite wired** — QSFR, EER, QGOM, M1_EFI, NLCS,
      and now CTR at any qubit count (see "Known limits").
- [x] **Cross-model H1-H5 statistical tests wired** —
      `scripts/run_hypothesis_tests.py`.
- [x] **Domain-based outlier filter (zero prices / >50% returns)**.
- [x] **PII sanitization gate wired**, with a real phone-regex
      false-positive bug found and fixed.
- [x] **Same-row target leakage fixed** (not just detected) —
      `src/evaluation/one_step_ahead.py`. Verified on a synthetic
      random walk: post-fix RMSE is ~2,500x worse than the persistence
      floor.
- [x] **QAR's clean-ASR baseline** —
      `src/attacks/adversarial.py`'s `compute_clean_asr()`.
- [x] **CTR (Coherence Time Retention) at any qubit count** —
      exact method (≤12 qubits) plus a new trajectory/Monte Carlo
      wavefunction method (`src/quantum/decoherence.py`) that scales
      past the ~16 TB dense-matrix ceiling. Includes a from-scratch
      mixed-state partial trace, validated against the closed-form
      Bell-state case. See "Known limits" above for the two real bugs
      caught and fixed during cross-validation, and the precision
      caveat on the fast default wired into `resilience_suite.py`.
- [x] **Repeated-runs experiment harness** —
      `scripts/run_repeated_experiment.py`.
- [x] **LLM wiring (zero-shot)** — `src/llm/llm_inference.py` +
      `src/baselines/llm_forecaster.py`.
- [x] **Graceful year-skipping instead of hard abort** —
      `_effective_end_year()` in `scripts/acquire_all_data.py`. A run
      before 2027 now fetches/consolidates through 2025 only, with a
      clear note, instead of refusing to run at all; automatically
      extends to 2026 with no code change once that year actually
      closes.
