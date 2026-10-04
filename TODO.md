# Codebase TODO — manuscript-vs-code remediation

Prioritized per the 09/28-10/02 conversation. Check off as each lands;
keep this file in sync with what's actually in the repo, not what's
planned.

**10 items waiting implementation** (everything under "Backlog" +
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
      `decide_skip_policy` UX). **This is now the single highest-value
      remaining item** — it's what actually makes the repeated-runs
      harness (below) tractable to run at the scale H1-H5 need.
- [ ] `run_all.py`: add `--override key=value` (repeatable) CLI flag,
      applied to the selected model's config after ablation-merging —
      needed by the startup prompt above to actually pass its choice
      down into a training run. (A narrower `--seed` override was added
      this session for the repeated-runs harness — see below — but the
      general-purpose flag is still open.)
- [ ] `run_all.py`: support `max_train_rows` / `max_val_rows` /
      `max_test_rows` per-model config keys — truncate to a
      chronological PREFIX (no shuffling — preserves time order for
      windowed models, and avoids introducing a new leakage vector).
- [ ] **No-grad fix in `qgan_llm.py`** — `_train_discriminator_step`
      calls `self.generator(z)` without `torch.no_grad()`, even though
      the result is `.detach()`'d before use. Add a
      `use_nograd_discriminator_pass` config flag (default True) so it's
      toggleable for direct before/after timing comparison, per the
      user's request. NOTE: this removes autograd/adjoint bookkeeping
      overhead only — NOT the dominant cost (the raw 20-qubit forward
      simulation itself still runs, since it's needed to produce the
      discriminator's fake-sample input regardless of whether its
      gradient is ever used). Don't oversell this as a fix for the
      "frozen at epoch 0" problem — the scale/ablation prompt above is
      what actually makes runs tractable.
      (Checked `qlstm_forecaster.py` for the same pattern: it already
      wraps both relevant calls in `torch.no_grad()` — no bug there.)

## Recommended, not yet started

- [ ] **Extend or disclose the cross-validation gap** — ForexSB/HistData
      cross-check is capped at 2023 (`XVAL_LAST_YEAR = min(2023, END_YEAR)`).
      2024-2026 (now 3+ years of the primary window) have zero independent
      corroboration of the Dukascopy series. Either extend the cross-check
      source or state this explicitly as a limitation.
- [ ] **Macro-feature staleness indicator** — GDP/CPI/etc. are forward-filled
      (correctly publication-lag-aware) but the model never sees *how old*
      a value is when it's using it (GDP can be up to 355 days old per the
      pipeline's own logged warning). Add a "days since last update" feature
      per macro series in `prepare_data.py`.
- [ ] **Reconsider outlier filtering (MAD 5.0/3.0) against stress periods** —
      a study whose hypotheses are about adversarial/noise robustness
      shouldn't casually filter out the real market-stress windows (2020
      COVID, 2022 rate-hike volatility) that those hypotheses are actually
      about. Check whether any known stress period got dropped; reconsider
      the threshold or carve out a stress-test subset instead of blanket
      filtering.
- [ ] **QCBM-based quantum discriminator** — the manuscript's own
      definition of QGANs is "quantum generator AND quantum
      discriminator," built from Quantum Circuit Born Machines. Current
      code: quantum generator, classical discriminator, no QCBM. Bigger
      lift than the others — a real architecture change.
- [ ] **Walk-forward / expanding-window CV**, replacing the single
      chronological train/val/test split — needed regardless of the
      k-fold claim (naive random-shuffle k-fold would leak future→past
      on this time series, so it has to be the walk-forward variant,
      not textbook k-fold).
- [ ] **M3->DV3 and M4->DV4 mediation** — `mediation_pipelines.py` has
      both wired as library functions (`scripts/run_hypothesis_tests.py`
      only calls M2 so far). M3 needs `fpr_per_run` computed per
      independent run (have per-model `fpr` now from
      `threat_detection.py`, but needs the repeated-runs harness below
      actually run to have multiple runs to pull from); M4 needs a
      `computational_overhead_ms` column (quantum-minus-classical
      latency delta) that doesn't exist in `results/all_results.csv` yet.

## Known limits of what landed this session (read before relying on these)

- **LLM Forecaster is zero-shot, not fine-tuned, and UNVERIFIED against
  a live endpoint.** `src/llm/llm_inference.py` and
  `src/baselines/llm_forecaster.py` are real, tested (against a mocked
  HTTP response — see below) integration code, registered in
  `run_all.py` only when `NVIDIA_API_KEY` is set. But this development
  session has no NVIDIA_API_KEY and no network route to api.nvidia.com
  at all (sandboxed egress allowlist), so the actual live call has
  never succeeded against a real account. Verify the request/response
  shape against NVIDIA's current API and your real model's behavior
  before trusting any number it produces — same caution
  `nvidia_finetune.py` already gave for the fine-tuning endpoint.
  Fine-tuning itself (`nvidia_finetune.submit_finetune_job` +
  `poll_finetune_status`) is a separate, hours-long, asynchronous step
  this model's `train()` does NOT run automatically — run it manually,
  then pass the resulting model_id via config to use a fine-tuned model
  here instead of a base one.
- **The repeated-runs harness (`scripts/run_repeated_experiment.py`) is
  real, resumable infrastructure — it does not, by itself, make 100-200
  runs per condition fast.** It's been tested for its own orchestration
  logic (state tracking, resumability, failure handling) with a mocked
  subprocess call, since this sandbox has no `data/processed/*.npy` to
  actually train against. Running it at the scale the manuscript's
  inferential statistics assume is still gated on the quantum-
  training-scale item above — without that, each QGAN-LLM-family run
  is the same severe bottleneck it's always been, just now repeatable
  100-200 times instead of once.
- **CTR now computes a real number, but only at ≤12 qubits.** This
  study's primary 20-qubit configuration needs ~1.1 TB for a dense
  density matrix — a fundamental simulation ceiling, not a missing
  feature. `resilience_suite.py` reports `ctr=NaN` with a clear reason
  above that threshold rather than guessing. The T1/T2 decoherence
  parameters in `src/quantum/decoherence.py` are simulation parameters,
  not measured hardware characteristics (no real quantum hardware is
  involved anywhere in this project) — report them alongside any CTR
  number.
- **Same-row leakage fix changes previously-reported RMSE/FID numbers.**
  Any `results/all_results.csv` row generated before this session's fix
  (`src/evaluation/one_step_ahead.py`) used the OLD, leaked same-row
  pairing for Classical GAN-LLM, QGAN-LLM, and QLSTM Forecaster — those
  numbers are not comparable to anything generated after this fix.
  Classical LSTM was never affected (its windowed target already used
  the correct convention) and needs no re-running on this account.

**Cross-checked again against the manuscript's actual H1-H5 text after
this session's wiring work:** all five hypotheses' dependent variables
are now PRODUCED by the pipeline on genuinely non-leaked data
(poisoning-resistance, threat-detection/FPR, FID, one-step-ahead RMSE,
ASR + a real clean-ASR baseline for QAR, entanglement entropy, mode
coverage, and now CTR at tractable qubit counts all compute and land in
`results/all_results.csv`; `scripts/run_hypothesis_tests.py` computes
the real H1-H5 statistics, QAR, and M2 mediation once that CSV has
rows). The remaining blocker for an actually valid conclusion is the
repeated-runs gap — every test still needs multiple independent
observations per condition, the harness now exists to produce them, but
hasn't been run at scale (see "Known limits" above) because the
training-scale bottleneck that would make that remotely tractable is
still the top item in Backlog.

## Deliberately NOT code work — manuscript text fixes instead

- Data Sanitization / differential privacy — out of scope for this
  study; note regex-only PII scanning as a limitation (the regex scan
  itself is now real and wired — see below — this item is specifically
  about NOT building differential privacy).
- GenAI "multi-modal" claim — scope down to tabular-only in the text.
- RMSE + decision-trees + k-fold conflation in Ch.1's definition — this
  reads like a definitions mix-up, not an implementation gap.
- Quantum-Classical Coherence vs. the code's separate "Coherence Time
  Retention" metric — just needs clarifying wording, not new code.
- **Single-broker/no-order-book-volume limitation** — Dukascopy's "volume"
  for FX is a broker-side proxy, not a consolidated tape. One
  limitations-section line, not code.
- Automated Vulnerability Assessment, Cyber Threat Intelligence, AI
  Guardrails, Adaptive Resilience/ADT — confirmed background/lit-review
  terms only; never claimed as implemented methodology or results in
  Ch.3/4. No code obligation, just make sure Ch.1 doesn't imply
  otherwise.
- **FPR mislabeled as "DV3" in an old code comment vs. Table 19's DV5** —
  found while wiring H5: `evaluation/metrics.py`'s `classification_metrics`
  docstring associates FPR with "DV3/H3," but Table 19 in the manuscript
  itself lists FPR as DV5, and H5's own text (not H3's) is the one about
  threat-detection accuracy. A manuscript-internal numbering
  inconsistency, not a code bug.

## Already resolved this conversation

- [x] `src/quantum/qaoa.py` (was the original crash — missing module).
- [x] `src/quantum/qpca.py` (QPCA via phase estimation + inverse QFT).
- [x] **TaLIS, real implementation** — `src/evaluation/talis.py`.
- [x] **Mode-collapse diagnostics** — `unique_sample_ratio`/`mode_coverage`,
      and FID/MMD/Wasserstein now actually reach `results/all_results.csv`.
- [x] **Dukascopy 2012 ingestion bug** (deterministic, not flaky) — fixed
      via `KNOWN_PROVIDER_HISTORY_START`; filenames renamed to match
      what's actually fetched.
- [x] **Extended study window to 15 years (2012-2026)**, with a
      `_refuse_if_incomplete_year()` guard applied to both Dukascopy and
      FRED acquisition.
- [x] **H3 wired end-to-end** (poisoning + model inversion) —
      `src/evaluation/poisoning_resistance.py`.
- [x] **H5 wired end-to-end** (threat detection / FPR) —
      `src/evaluation/threat_detection.py` (interim classical detector;
      swap for `llm/threat_scoring.py` once the LLM Forecaster's output
      is trusted enough to build a text-based detector around).
- [x] **Quantum resilience suite wired** — QSFR, EER, QGOM, M1_EFI, NLCS
      via `src/evaluation/resilience_suite.py`.
- [x] **Cross-model H1-H5 statistical tests wired** —
      `scripts/run_hypothesis_tests.py` (ANCOVA/t-test/Pearson/M2
      mediation/QAR/CQRS).
- [x] **Domain-based outlier filter (zero prices / >50% returns)** —
      `domain_implausible_mask()` in `src/data/preprocessing.py`, the
      missing first pass of Ch.3's two-pass cleaning protocol.
- [x] **PII sanitization gate wired** —
      `scripts/scan_run_artifacts_for_pii.py`; found and fixed a real
      false-positive bug in the `phone` regex while testing it.
- [x] **Same-row target leakage fixed** (not just detected) —
      `src/evaluation/one_step_ahead.py`'s `shift_for_one_step_ahead()`,
      applied locally in `ClassicalGANLLM`/`QGANLLM`/`QLSTMForecaster`'s
      `train()`/`evaluate()`. Deliberately NOT fixed by changing the
      shared `X_*.npy`/`y_*.npy` files — Classical LSTM's
      `WindowedSequenceDataset` depends on the original same-row
      convention for ITS OWN correctness (its window never includes the
      target row, so there was never leakage for LSTM specifically).
      Verified on a synthetic random walk (by construction unforecastable
      from same-row features): post-fix RMSE is ~2,500x worse than the
      one-step persistence floor, confirming the model can no longer read
      the target out of its own input. Permanent regression test in
      `tests/test_one_step_ahead_leakage_fix.py`.
- [x] **QAR's clean-ASR baseline** —
      `src/attacks/adversarial.py`'s `compute_clean_asr()` (epsilon=0
      FGSM/PGD; Carlini-Wagner has no equally direct zero-perturbation
      setting, so it's excluded from the clean baseline, documented).
      Saved as `asr_clean` by all three same-row-fixed baselines; QAR
      wired into `scripts/run_hypothesis_tests.py`.
- [x] **CTR (Coherence Time Retention)** — `src/quantum/decoherence.py`,
      a genuine decoherence-channel time-series simulator (PennyLane's
      `default.mixed` device; the only module here that uses it) that
      didn't exist anywhere before. Includes a from-scratch mixed-state
      partial-trace implementation (the existing
      `tomography.reduced_density_matrix` only handles pure states),
      validated against the closed-form Bell-state case (reduces to I/2
      exactly). Wired into `resilience_suite.py`, capped at the existing
      12-qubit tractability ceiling (NaN with a clear reason above that,
      not a crash or a guess).
- [x] **Repeated-runs experiment harness** —
      `scripts/run_repeated_experiment.py`: runs a named model config N
      times across seeds, resumable via a state file, appending each run
      to `results/all_results.csv` with a `seed` column (new
      `run_all.py --seed` flag) so `run_hypothesis_tests.py` can consume
      the result with no changes. Infrastructure only — see "Known
      limits" above for what it doesn't solve by itself.
- [x] **LLM wiring (zero-shot)** — `src/llm/llm_inference.py` (NVIDIA
      chat-completions call, prompt building, response parsing) and
      `src/baselines/llm_forecaster.py` (the first model in this
      pipeline whose forecast actually comes from an LLM). Registered in
      `run_all.py` only when `NVIDIA_API_KEY` is set, so it doesn't break
      anyone else's run. See "Known limits" above for what's NOT done
      (fine-tuning itself, live-endpoint verification).
