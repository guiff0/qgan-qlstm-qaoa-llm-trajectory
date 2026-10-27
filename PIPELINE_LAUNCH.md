# PIPELINE_LAUNCH.md — running the whole project with one command

`run_pipeline.py`, at the repo root, is the single entry point for the
whole project: environment check → data acquisition → data preparation →
train + evaluate every model → Chapter 4 comparison. This replaces running
six separate scripts by hand in the right order (RUNNING.md still documents
each of those scripts individually, for when you want to run just one).

```bash
python run_pipeline.py
```

## What it actually runs, in order

| # | Step id | Command it runs |
|---|---|---|
| 1 | `verify_environment` | `python -m scripts.verify_environment` |
| 2 | `acquire_dukascopy` | `python -m scripts.acquire_all_data --steps dukascopy` |
| 3 | `acquire_xval` | `python -m scripts.acquire_all_data --steps xval` |
| 4 | `acquire_fred` | `python -m scripts.acquire_all_data --steps fred` |
| 5 | `prepare_data` | `python -m scripts.prepare_data` |
| 6 | `train_eval:<model>` | `python -m src.experiments.run_all --only "<model>"` — one per baseline + config-defined ablation |
| 7 | `verify_chapter4` | `python -m scripts.verify_chapter4` |

Steps 2–4 are independent of each other; steps 6 (one per model) are
independent of each other. `prepare_data` and `verify_chapter4` are hard
barriers — they need every step before them to have finished.

## Resumability: the whole point

Every stage that finishes gets recorded in `data/pipeline_state.json`
(status, timestamp, duration, and whether its output file is actually
still on disk). If the process is killed — crash, Ctrl-C, a disconnected
SSH session, an OOM kill in the middle of a multi-hour quantum training
run — running `python run_pipeline.py` again picks up **after the last
step that actually finished**, instead of re-paying for everything from
the start.

A step is only ever skipped if **both** are true: the state file says
`"completed"`, **and** its expected output file is still there. A
`"completed"` marker whose file has since been deleted is never trusted —
that step re-runs.

### The one prompt

If (and only if) there is something it could skip, `run_pipeline.py` asks
**exactly once**, at the very start:

```
Found 6 previously completed step(s) from an earlier run of this pipeline:
  - acquire_dukascopy            completed 2026-09-26T06:55:25Z (0.9s)
  - acquire_fred                 completed 2026-09-26T06:55:25Z (0.4s)
  - prepare_data                 completed 2026-09-26T06:55:28Z (3.1s)
  - train_eval:Classical LSTM    completed 2026-09-26T06:56:16Z (48.1s)
  - train_eval:Classical GAN-LLM completed 2026-09-26T06:59:01Z (48.8s)
  - verify_chapter4              completed 2026-09-26T06:59:01Z (0.3s)

Skip these and continue where you left off? [Y/n]:
```

That one answer applies to every step for the rest of the run — it never
asks again per-step. On a fresh repo with no prior state, nothing is
skippable, so it asks nothing and just starts.

In a non-interactive session (no TTY — a cron job, a CI runner, a piped
command) with neither `--yes` nor `--no-skip` given, it defaults to
**skip** and says so explicitly, rather than hanging on `input()`.

## Every flag

```
python run_pipeline.py                          # interactive (asks once, if relevant)
python run_pipeline.py --yes                     # non-interactive: auto-skip completed stages
python run_pipeline.py --no-skip                 # non-interactive: redo everything, ignore state
python run_pipeline.py --reset                   # delete the state file first, then run everything
python run_pipeline.py --only "Classical LSTM,QGAN-LLM"   # subset of models for step 6
python run_pipeline.py --reuse-checkpoints       # passed through to run_all for models that support it
python run_pipeline.py --status                  # print the state file as JSON and exit; runs nothing
python run_pipeline.py --jobs 3                  # parallelize within the acquire and train_eval groups
python run_pipeline.py --watch                   # attach to an already-running pipeline; see below
python run_pipeline.py --watch --watch-interval 5
python run_pipeline.py --config path/to/other_config.yaml
```

`--yes` and `--no-skip` are mutually exclusive.

## Parallelism: `--jobs N`

`--jobs 0` (the default) picks the parallelism automatically (acquire group: all at once; train_eval group: min(cores/2, free RAM/3 GB, number of models), with BLAS/torch threads divided among jobs). `--jobs 1` is fully sequential — identical to running each
script by hand in order, one at a time.

`--jobs N` (N > 1) runs the independent steps **within** the acquire
group and **within** the train_eval group up to N at a time.
`prepare_data` and `verify_chapter4` always run alone, since they are
hard barriers.

```
verify_environment (solo)
        |
   ┌────┼────┐
dukascopy  xval  fred          <- parallelizable with --jobs
   └────┼────┘
   prepare_data                 <- barrier: needs all three
        |
   ┌────┼────┬─────┬─────┐
 LSTM  GAN  QGAN  QLSTM  ...    <- parallelizable with --jobs
   └────┼────┴─────┴─────┘
   verify_chapter4              <- barrier: needs every model's result
```

**Caveat on latency:** latency is timed inside each job, so parallel runs inflate it. Results record `concurrent_jobs`; for a latency number worth reporting, re-run that model alone with `--jobs 1`.

**Caveat:** the quantum models (`QGAN-LLM`, `QLSTM Forecaster`, and their
ablations) all run on PennyLane's `default.qubit` CPU simulator. Running
several of those at once competes for the same CPU cores rather than
truly parallelizing — `--jobs` helps most for the acquire group (network-
bound, cheap) and for mixing classical + quantum training together, less
so for running multiple quantum models side by side on a small machine.

When a parallel batch is running, output is redirected to
`logs/pipeline/<step_id>.log` per step instead of the shared terminal
(several tqdm bars writing to one terminal at once garbles the output) —
see Central monitoring below for how to actually watch progress in that
case.

**Concurrency safety:** every model in a `--jobs`-parallelized train_eval
batch appends its own row to the same `results/all_results.csv`. This is
protected by a cross-process file lock (`src/utils/io.py`'s `file_lock`,
used inside `RunLogger.append_to_results_csv`) — verified under a forced
race window that two processes finishing at the same moment do not
silently clobber each other's row (see `tests/test_file_lock.py`).

## Central monitoring

You never have to go hunting through `logs/pipeline/*.log` by hand to see
whether a parallel batch is still making progress:

- **Automatically, in the same terminal:** while a `--jobs > 1` batch is
  running, a live table redraws every few seconds showing every step in
  that batch — status, elapsed time, and the latest progress line pulled
  from that step's own log (batch counts, loss values, whatever it last
  printed).

  ```
  ==============================================================================
   PIPELINE STATUS  14:38:49 UTC
  ------------------------------------------------------------------------------
   [RUN ] train_eval:Classical LSTM            20s  epoch 0 batches: 1%|...
   [RUN ] train_eval:Classical GAN-LLM         20s  epoch 0 batches: 0%|...
  ==============================================================================
  ```

- **From a second terminal or SSH session:** `python run_pipeline.py
  --watch` shows the exact same table for a pipeline that's running (or
  finished) elsewhere. It only reads `data/pipeline_state.json` and the
  per-step log files — it never starts, stops, or otherwise affects the
  running pipeline. It exits automatically once every step has reached a
  final state (`completed` or `failed`), or on Ctrl-C.

  ```
  python run_pipeline.py --watch --watch-interval 5
  ```

Sequential (`--jobs 1`) steps don't need any of this — their full output,
including every progress bar, streams straight to the terminal exactly as
if you'd run that script yourself.

## The state file

`data/pipeline_state.json` — plain, readable JSON, so you can open it,
see exactly what ran and when, or hand-edit/delete an entry to force a
step to be considered incomplete without deleting its actual output.

```json
{
  "schema_version": 1,
  "created_utc": "2026-09-26T06:55:17Z",
  "updated_utc": "2026-09-26T06:59:13Z",
  "steps": {
    "acquire_dukascopy": {
      "status": "completed",
      "completed_utc": "2026-09-26T06:55:25Z",
      "duration_seconds": 0.9,
      "detail": "artifact verified"
    },
    "train_eval:Classical GAN-LLM": {
      "status": "running",
      "started_utc": "2026-09-26T06:55:28Z"
    }
  }
}
```

A step stuck at `"status": "running"` after a crash is exactly the signal
that it needs to be retried on the next invocation — and it will be,
automatically, because `"running" != "completed"`.

## Failure handling

- **`verify_environment` failures never abort the pipeline.** It exits
  nonzero on *any* failed check, including ones irrelevant to what you're
  about to run (e.g. a missing `NVIDIA_API_KEY` when nothing here touches
  the LLM step). Each downstream step still fails loudly and specifically
  on its own if something it truly needs is missing.
- **`acquire_xval` (the HistData.com cross-check) is optional** — its own
  script never raises on failure, so a failure here never blocks anything
  downstream.
- **A required step failing (`acquire_dukascopy`, `acquire_fred`,
  `prepare_data`) aborts the pipeline immediately.** Fix the error shown
  and re-run; everything already completed is skipped automatically.
- **One model failing in the train_eval group does not stop the others.**
  Models are independent, so the pipeline keeps going, collects every
  failure, and still runs `verify_chapter4` at the end against whatever
  models did succeed. The final exit code is nonzero if anything failed,
  and the failed model names are printed so you know exactly what to
  retry.

## Testing

`run_pipeline.py`'s own logic (state persistence, the once-only skip
prompt, parallel batching, failure handling, the dashboard renderer) is
covered by `tests/test_run_pipeline.py` (33 tests) with every subprocess
call mocked — these test the orchestrator itself, not the real scripts it
invokes (those have their own test files: `test_data_ingestion.py`,
`test_standalone_demo.py`, etc.). The cross-process file lock has its own
dedicated tests in `tests/test_file_lock.py`, including one that
deterministically forces the race window open to prove data loss
happened *before* the fix and does not happen *after* it.

```bash
python -m pytest tests/test_run_pipeline.py tests/test_file_lock.py -v
```
