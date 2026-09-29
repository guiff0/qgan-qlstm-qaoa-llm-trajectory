# Codebase TODO — manuscript-vs-code remediation

Prioritized per the 09/28 conversation. Check off as each lands; keep
this file in sync with what's actually in the repo, not what's planned.

**8 items waiting implementation** (everything under "Backlog" +
"Recommended, not yet started" below). "Deliberately not code work" is
a separate list — those are manuscript-text fixes, not implementation
gaps, so they don't count toward that number.

## Backlog — not started

- [ ] **LLM wiring** — route an actual forecast through
      `src/llm/nvidia_finetune.py` instead of the `nn.Linear` head, OR
      explicitly scope the manuscript down to "linear head, LLM
      fine-tuning client available but not in the critical path."
- [ ] **Startup prompt for quantum training scale** — `run_pipeline.py`
      should ask the user, once per fresh run: which qubit
      ablation(s) to include (20q ring baseline / 12-qubit /
      linear-topology / no-noise / QLSTM Forecaster), a training-scale
      preset (rows subsampled + epochs — the full 3.7M-row/58k-batch
      schedule is not tractable for a 20-qubit lightning.qubit circuit
      on a single CPU machine), and whether to apply the no-grad fix
      below. Persist the choice in the pipeline's state file so a
      resumed run doesn't re-prompt (matches the existing
      `decide_skip_policy` UX).
- [ ] `run_all.py`: add `--override key=value` (repeatable) CLI flag,
      applied to the selected model's config after ablation-merging —
      needed by the startup prompt above to actually pass its choice
      down into a training run.
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
- [ ] **Wire FPR into the pipeline** — confirmed (not just suspected):
      `classification_metrics` in `src/evaluation/metrics.py` computes
      FPR correctly, but nothing in `run_all.py` or any baseline's
      `evaluate()` calls it. It needs a binary threat/benign label
      source to run against — that labeling scheme doesn't exist yet
      either, so this is two pieces of work, not one.

## Deliberately NOT code work — manuscript text fixes instead

- Data Sanitization / differential privacy — out of scope for this
  study; note regex-only PII scanning as a limitation.
- GenAI "multi-modal" claim — scope down to tabular-only in the text.
- RMSE + decision-trees + k-fold conflation in Ch.1's definition — this
  reads like a definitions mix-up, not an implementation gap.
- Quantum-Classical Coherence vs. the code's separate "Coherence Time
  Retention" metric — just needs clarifying wording, not new code.
- Automated Vulnerability Assessment, Cyber Threat Intelligence, AI
  Guardrails, Adaptive Resilience/ADT — confirmed background/lit-review
  terms only; never claimed as implemented methodology or results in
  Ch.3/4. No code obligation, just make sure Ch.1 doesn't imply
  otherwise.

## Already resolved this conversation

- [x] `src/quantum/qaoa.py` (was the original crash — missing module).
- [x] `src/quantum/qpca.py` (QPCA via phase estimation + inverse QFT).
- [x] **TaLIS, real implementation** — `src/evaluation/talis.py`,
      wired into `run_all.py`'s per-model metrics, config in
      `config/default_config.yaml`, tests. This was the manuscript's
      own claimed novel contribution with zero prior implementation.
- [x] **Mode-collapse diagnostics** — `unique_sample_ratio` and
      `mode_coverage` added to `src/evaluation/metrics.py`'s
      `synthetic_data_fidelity_report`, which is now actually CALLED
      from both `QGANLLM.evaluate()` and `ClassicalGANLLM.evaluate()`
      (previously existed but was only exercised by tests — FID/MMD/
      Wasserstein weren't reaching `results/all_results.csv` either;
      that's fixed too as a side effect).
- [x] **Dukascopy 2012 ingestion bug** (deterministic, not flaky) —
      `_validate_year` unconditionally required every year to start by
      Jan 8, but Dukascopy's real free EURUSD 1-min history only goes
      back to 2012-01-11, so 2012 was rejected as "truncated" on every
      single run. Fixed via `KNOWN_PROVIDER_HISTORY_START` (evidence-
      based: this project's own prior successful logs confirmed the
      2012-01-11 start), with the row-count floor scaled proportionally
      for that year too. Also fixed: `data.dukascopy_file` /
      `forexsb_file` / `fred_file` were named "..._2010_..." even
      though `train_start=2012-01-01` means 2010/2011 were never
      fetched at all — renamed to match what's actually requested
      (2012/2012/2011 respectively) rather than widening the real study
      window to match the old (wrong) name. Regression tests added in
      `tests/test_acquire_all_data.py`.
