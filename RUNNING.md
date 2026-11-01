# RUNNING.md — Executing the Pipeline

Complete `SETUP.md` first. This document assumes your data files are in
place and `python -m scripts.verify_environment` passes (data-file checks
aside, until Step 1 below).

## Step 0 — Smoke test (seconds, no data needed)

```bash
pytest tests/test_smoke.py -v
```
All 8 tests should pass. This confirms the *code* works — it does not
and cannot confirm the dissertation's numbers, since it runs on tiny
synthetic data. If anything fails here, stop and fix it before spending
hours on a real run; something is broken in a way that has nothing to do
with data or compute scale.

## Step 1 — Prepare the data (minutes to ~1 hour, mostly I/O bound)

```bash
python -m scripts.prepare_data
```

This loads the raw CSVs, merges them (forward-filling daily VIX/FRED data
onto minute-level timestamps), cleans outliers via MAD (5 MADs for
prices, 3 for indicators — see `src/data/preprocessing.py`), engineers
the 10 technical indicators, fits PCA on the training split only, and
writes `data/processed/{X,y}_{train,val,test}.npy`.

**Check the printed row counts against your expectations** before
proceeding — if a split reports 0 rows, your date ranges in
`config/default_config.yaml` (`data.train_start` through
`data.test_end`) don't overlap with what's actually in your raw files.

Re-run `python -m scripts.verify_environment` — the "Processed file
present" checks should now all pass.

## Step 2 — Run the experiments (hours to ~1–2 days, GPU-bound)

```bash
python -m src.experiments.run_all
```

Trains and evaluates, in order: Classical LSTM, Classical GAN-LLM,
QGAN-LLM (primary, 20-qubit), QLSTM Forecaster (the quantum circuit used
directly as the forecaster, no GAN wrapper — see
`src/baselines/qlstm_forecaster.py`'s module docstring for how this
differs from QGAN-LLM and why its latency numbers are not directly
comparable to `ClassicalLSTM`'s), then every ablation listed under
`ablations:` in the config (12-qubit, linear-topology, no-noise, by
default). Each model's full training history, final metrics, and
environment fingerprint are written to `logs/<run_id>.log` and
`logs/<run_id>.json`; one summary row per model is appended to
`results/all_results.csv`.

**Expected runtime** (rough, CPU-only quantum simulation, GPU for the
classical components; scale up if running everything CPU-only):
- Classical LSTM: tens of minutes at 5.5M rows with real windowing.
- Classical GAN-LLM: similar order of magnitude.
- QGAN-LLM (20-qubit, `default.qubit`): this is the long pole. Each
  forward pass of the quantum circuit simulates a 2^20-dimensional state
  vector; at realistic epoch counts and batch sizes this is genuinely a
  multi-hour-to-multi-day run on CPU. Consider:
  - `qgan_llm.quantum_device: "lightning.qubit"` in the config (faster
    CPU backend, same physics, installed via `pennylane-lightning` which
    is already in `requirements.txt`).
  - Running a shorter `epochs` count first as a sanity check on
    real data before committing to a full run.
  - Running ablations in parallel on separate machines/processes if you
    have the hardware — `--only "QGAN-LLM (12-qubit)"` etc. lets you
    split the work (see below).
- QLSTM Forecaster: the *quantum circuit itself* runs on every single
  training and inference call here (unlike QGAN-LLM, whose quantum
  circuit is only invoked during training's data-augmentation step, not
  at inference — see that file's docstring). Expect this baseline's
  per-sample latency to be visibly, genuinely higher than every other
  baseline's, not just at training time. That's expected and is in fact
  the point of including it: it is the one baseline where a latency
  comparison actually measures quantum-circuit inference cost.

**To run a subset** (e.g. re-running just one model after a crash, or
splitting work across machines):
```bash
python -m src.experiments.run_all --only "QGAN-LLM"
python -m src.experiments.run_all --only "Classical LSTM,Classical GAN-LLM"
```

**To use a different config** (e.g. a scaled-down version for a first
pass):
```bash
python -m src.experiments.run_all --config config/my_variant.yaml
```

If the process is killed partway through (timeout, OOM, disconnect),
already-completed models' rows remain in `results/all_results.csv` —
re-run with `--only` naming just the remaining models rather than
starting over.

## Step 3 — LLM fine-tuning (separate, hours, needs NVIDIA_API_KEY)

The baseline/QGAN training loops in Step 2 do NOT themselves call the
NVIDIA fine-tuning API — that's a separate, long-running job better
suited to a background process than an interactive script. After Step 2
has produced synthetic data via a trained generator:

```python
from src.llm.nvidia_finetune import build_training_examples, submit_finetune_job, poll_finetune_status, save_job_manifest

examples = build_training_examples(real_texts, synthetic_texts, real_ratio=0.6, synthetic_ratio=0.4)
job = submit_finetune_job(examples, base_model="meta/llama-3.3-70b-instruct", epochs=3)
save_job_manifest(job)
status = poll_finetune_status(job["id"])  # blocks until done; run this in a background process, e.g. `nohup python your_script.py &`
```

`real_texts`/`synthetic_texts` (the text serialization of your
real/QGAN-synthetic market feature vectors into LLM fine-tuning examples)
aren't generated by any file here — the manuscript doesn't specify an
exact serialization format (e.g. how a row of OHLCV + indicators becomes
a training prompt), so this is a real design decision for you to make and
document, not something safe to guess at silently.

> **Order matters: `run_all` first.** `results/all_results.csv` is written only by `src.experiments.run_all`
> (directly, via `run_pipeline.py`, or via `scripts.run_repeated_experiment`, which calls it per seed).
> `verify_chapter4` and `run_hypothesis_tests` only read that file, and both now stop with exit code 2 and a
> pointer back here if it is missing or empty.
>
> 1. `python -m src.experiments.run_all`
> 2. `python -m scripts.verify_chapter4`
> 3. `python -m scripts.run_hypothesis_tests`

## Step 4 — Verify against the manuscript's claims

```bash
python -m scripts.verify_chapter4
```

Edit `DISSERTATION_CLAIMS` at the top of `scripts/verify_chapter4.py` to
whatever specific numbers you want checked (it ships with the disputed
figures from earlier review pre-filled as a starting point — replace or
extend them as needed). This prints every comparison — model, metric,
claimed value, actual value, percent difference — not just the ones that
fail, and writes a reshaped `results/chapter4_numbers.json` for pulling
into the manuscript's actual tables.

## What to send back

For a result to be checked against Chapter 4, the most useful things to
send back are:

1. **`results/all_results.csv`** — the one-row-per-model summary.
2. **`results/chapter4_numbers.json`** — the reshaped comparison tables.
3. **The full `logs/` directory** (or at minimum, the `.json` log for
   each model) — this has the complete training history and, for models
   with latency measured, the full per-sample latency distribution (not
   just mean/SD), which is specifically what's needed to check a claim
   like "97% of runs fall below 40ms" against reality rather than
   assuming a normal distribution from mean/SD alone.
4. **The console output of `python -m scripts.verify_chapter4`** —
   already does the comparison math for you.
5. If anything crashed or produced results you're unsure about, the
   **full traceback and the exact command you ran** — a partial log with
   an error is more useful than no log at all.

Anything in `logs/*.json` also includes an environment fingerprint
(package versions, git commit if this is in a git repo, timestamp) so
results can be tied to exactly what code and environment produced them.
