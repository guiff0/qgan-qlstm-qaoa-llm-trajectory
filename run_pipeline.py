#!/usr/bin/env python3
"""
================================================================================
run_pipeline.py -- single entry point for the whole QGAN-LLM pipeline
================================================================================
GOAL
    Run every stage of the project -- environment check, data acquisition,
    data preparation, training + evaluation of every model, and the Chapter 4
    comparison -- with ONE command, instead of running six separate scripts
    by hand in the right order (see RUNNING.md for what this replaces).

OBJECTIVE
    Make a multi-hour, multi-stage pipeline SAFE TO INTERRUPT AND RESUME.
    Every stage that finishes is recorded in a small state file; if the
    process is killed (crash, Ctrl-C, disconnected SSH session, out-of-
    memory kill) and you run this again, it picks up after the last stage
    that actually completed, rather than re-running -- and re-paying for --
    everything from the start.

ASSUMPTIONS
    - You have already followed SETUP.md (dependencies installed, raw data
      files or API credentials in place) at least up to the point where
      `python -m scripts.verify_environment` can meaningfully run.
    - config/default_config.yaml reflects the run you want (model
      hyperparameters, ablations, date ranges). This script does not
      change your config; it only decides WHICH scripts to invoke and
      WHETHER to skip ones already done.
    - Each stage below is itself idempotent and safe to re-run (this was
      true of every script this orchestrates before this file existed);
      this script adds resumability on top, it does not change what any
      individual stage does.
    - Running as `python run_pipeline.py` from the repository root (same
      assumption every other script here already makes).

EXECUTION STEPS (in order; see STEPS below for the authoritative list)
    1. verify_environment  -- python -m scripts.verify_environment
    2. acquire_dukascopy   -- python -m scripts.acquire_all_data --steps dukascopy
    3. acquire_xval        -- python -m scripts.acquire_all_data --steps xval
    4. acquire_fred        -- python -m scripts.acquire_all_data --steps fred
    5. prepare_data        -- python -m scripts.prepare_data
    6. train_eval:<model>  -- python -m src.experiments.run_all --only "<model>"
                              (one invocation per model/ablation in your
                              config, so a single model's crash doesn't
                              lose progress on the others)
    7. verify_chapter4     -- python -m scripts.verify_chapter4

WHY SUBPROCESSES, NOT DIRECT FUNCTION CALLS
    Each stage is invoked as a separate `python -m ...` subprocess, exactly
    as documented in RUNNING.md, rather than importing and calling each
    script's main() in-process. Two reasons: (1) memory from a finished
    stage (a multi-GB pandas DataFrame in prepare_data, a trained model's
    CUDA/CPU tensors) is fully released by the OS when that subprocess
    exits, instead of accumulating across a many-hour single process; (2)
    each stage's own progress bars, prints, and logs pass straight through
    unmodified -- this file adds resumability without duplicating or
    reimplementing any stage's own logic.

THE "ASK ONCE" RULE (see the docstring on `decide_skip_policy` below)
    This script prompts at most ONCE per invocation, at the very start, and
    only if there is something it COULD skip. That single answer ("skip
    everything the state file says is done" / "ignore the state file and
    redo everything") is then applied uniformly to every stage for the rest
    of this run -- it does not ask again stage by stage.

CENTRAL MONITORING (see `render_dashboard` / `watch_pipeline` below)
    This same file is also the monitoring surface for the whole run, not
    just the launcher: while a --jobs>1 parallel batch is in flight, it
    automatically redraws a status table for every step in that batch
    (status, elapsed time, and the latest progress line from that step's
    own log) every few seconds -- so you never need to go hunting through
    logs/pipeline/*.log by hand just to see whether things are still
    moving. The exact same table is available on demand, from a second
    terminal or SSH session, via `python run_pipeline.py --watch`, which
    reads the same state file and log files the running pipeline is
    already writing and never starts or affects anything itself.

Run:
    python run_pipeline.py                  # interactive (asks once, if relevant)
    python run_pipeline.py --yes            # non-interactive: auto-skip completed stages
    python run_pipeline.py --no-skip        # non-interactive: redo everything, ignore state
    python run_pipeline.py --reset          # wipe the state file first, then run everything
    python run_pipeline.py --only "Classical LSTM,QGAN-LLM"   # subset of models for step 6
    python run_pipeline.py --status         # print the state file and exit; runs nothing
    python run_pipeline.py --jobs 3         # run the acquire group, then the train_eval group,
                                             # up to 3 steps at a time within each (see --help)
    python run_pipeline.py --watch          # from another terminal: live status of a running pipeline
    python run_pipeline.py --stats-only     # ONLY run_hypothesis_tests (no env check, no model listing)
    python run_pipeline.py --analysis-only  # ONLY verify_chapter4 + run_hypothesis_tests
    python run_pipeline.py --steps run_hypothesis_tests,scan_logs_for_pii   # any explicit step ids
    python run_pipeline.py --extra-seeds 2 --seed-models "Classical GAN-LLM,QGAN-LLM (no noise)"
                                             # independent repeats (seeds 43, 44) so H1/H3 have n >= 2

ANALYSIS STEPS (verify_chapter4, run_hypothesis_tests) ARE NEVER SKIPPED AND NEVER BLOCKED
    They read results/all_results.csv, which changes with every run, so a stored "completed"
    (or "failed") marker means nothing for them: they re-run on EVERY invocation, whatever
    --yes / --no-skip / the prompt answer was. A failed train_eval step, or a failure in an
    earlier required step, does not stop them (they are cheap and read whatever results exist),
    and a failure in one analysis step does not stop the other.
================================================================================
"""
from __future__ import annotations

import time as _time


def _ts():
    return _time.strftime("[%Y-%m-%d %H:%M:%S]")


import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config  # noqa: E402
from src.utils.io import atomic_write_text  # noqa: E402

# ------------------------------------------------------------------------
# The placeholder / resume-state file.
#
# Lives under data/ (not logs/ or results/) because it describes what the
# DATA PIPELINE has produced and validated -- it sits next to data/raw/ and
# data/processed/, the directories whose contents it is tracking, so
# "where did my run get to" and "where is my data" are answered by looking
# in the same place. It is plain, readable JSON (not a database or a lock
# file), specifically so a person can open it, see exactly what ran, when,
# and how, and hand-edit or delete it if they ever want to force a stage
# to be considered incomplete without deleting the actual output files.
# ------------------------------------------------------------------------
STATE_PATH = os.path.join("data", "pipeline_state.json")
STATE_SCHEMA_VERSION = 1
# Parallel steps run in threads and all mutate/save the same state dict.
# RLock so mark_* can hold it while save_state re-acquires it.
_state_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> dict:
    if not os.path.isfile(STATE_PATH):
        return {"schema_version": STATE_SCHEMA_VERSION, "created_utc": _now(), "steps": {}}
    with open(STATE_PATH) as f:
        state = json.load(f)
    state.setdefault("steps", {})
    return state


def save_state(state: dict) -> None:
    """Atomic write (temp file + os.replace, via src/utils/io.py) after
    EVERY step, not just at the end -- so if the process is killed mid-
    pipeline, the state file on disk always reflects every step that
    genuinely finished, never a partially-written or stale file."""
    from src.utils.run_identity import pipeline_run_id
    with _state_lock:
        state["updated_utc"] = _now()
        state["pipeline_run_id"] = pipeline_run_id()
        state.setdefault("pipeline_runs", [])
        if state["pipeline_run_id"] not in state["pipeline_runs"]:
            state["pipeline_runs"].append(state["pipeline_run_id"])
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        atomic_write_text(STATE_PATH, json.dumps(state, indent=2, default=str))


def mark_running(state: dict, step_id: str) -> None:
    with _state_lock:
        state["steps"][step_id] = {"status": "running", "started_utc": _now()}
        save_state(state)


def mark_completed(state: dict, step_id: str, duration_s: float, detail: str = "") -> None:
    with _state_lock:
        state["steps"][step_id] = {
            "status": "completed",
            "completed_utc": _now(),
            "duration_seconds": round(duration_s, 1),
            "detail": detail,
        }
        save_state(state)


def mark_failed(state: dict, step_id: str, duration_s: float, returncode: int) -> None:
    with _state_lock:
        state["steps"][step_id] = {
            "status": "failed",
            "failed_utc": _now(),
            "duration_seconds": round(duration_s, 1),
            "returncode": returncode,
        }
        save_state(state)


# ------------------------------------------------------------------------
# Step definitions: each step is (step_id, description, command, artifact
# check). `artifact_ok` is a defensive second opinion -- the state file
# might say "completed" but if the file it produced is gone (data/raw/
# cleared, a .npy deleted by hand, etc.), that "completed" marker is
# stale and must not be trusted. A step both marked completed AND passing
# its artifact check is the only thing eligible to be skipped.
# ------------------------------------------------------------------------
def _nonempty(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0


def build_steps(cfg: dict, model_names: list[str], only_models: list[str] | None,
                extra_seeds: list[tuple[str, int]] | None = None) -> list[dict]:
    data_cfg = cfg["data"]
    processed = data_cfg["processed_dir"]

    steps = [
        {
            "id": "verify_environment",
            "desc": "Verify environment (packages, config, data files)",
            "cmd": [sys.executable, "-m", "scripts.verify_environment"],
            # No unique artifact of its own -- it is cheap (seconds) to
            # re-run, so this check always reports "not satisfied", i.e.
            # this step re-runs on every invocation regardless of the skip
            # policy. That's intentional: it's a live check of the CURRENT
            # environment, and a stale "it passed last time" from a prior
            # session/venv is exactly the thing worth NOT trusting silently.
            "artifact_ok": lambda: False,
            # ADVISORY, NOT A HARD GATE: verify_environment.py exits nonzero
            # on ANY failed check, including ones irrelevant to the stages
            # you're actually about to run here (e.g. NVIDIA_API_KEY is only
            # needed for the LLM fine-tuning step, which this pipeline never
            # invokes; dukascopy_python/histdata are only needed if the raw
            # files aren't already on disk). Tested against a real failure
            # during development: a single missing optional package aborted
            # the entire pipeline before touching any actual data. Each
            # downstream step below still fails loudly and specifically on
            # its own (e.g. prepare_data.py raises DataFileNotFoundError
            # with exact remediation steps) if something it truly needs is
            # missing, so this stays a warning rather than a blocker.
            "required": False,
        },
        {
            "id": "acquire_dukascopy",
            "desc": "Acquire Dukascopy EUR/USD 1-min OHLCV",
            "cmd": [sys.executable, "-m", "scripts.acquire_all_data", "--steps", "dukascopy"],
            "artifact_ok": lambda: _nonempty(data_cfg["dukascopy_file"]),
            "parallel_group": "acquire",
        },
        {
            "id": "acquire_xval",
            "desc": "Acquire HistData.com cross-check series (supplementary)",
            "cmd": [sys.executable, "-m", "scripts.acquire_all_data", "--steps", "xval"],
            # Supplementary: acquire_crosscheck() itself never raises on
            # failure (see scripts/acquire_all_data.py), so "ran" and
            # "produced the file" are different things here on purpose.
            "artifact_ok": lambda: _nonempty(data_cfg["forexsb_file"]),
            "required": False,
            "parallel_group": "acquire",
        },
        {
            "id": "acquire_fred",
            "desc": "Acquire FRED macro series",
            "cmd": [sys.executable, "-m", "scripts.acquire_all_data", "--steps", "fred"],
            "artifact_ok": lambda: _nonempty(data_cfg["fred_file"]),
            "parallel_group": "acquire",
        },
        {
            "id": "prepare_data",
            "desc": "Merge sources, engineer features, split, scale, PCA, export arrays",
            "cmd": [sys.executable, "-m", "scripts.prepare_data"],
            "artifact_ok": lambda: all(
                _nonempty(os.path.join(processed, f"{prefix}_{split}.npy"))
                for prefix in ("X", "y") for split in ("train", "val", "test")
            ),
        },
    ]

    names = only_models if only_models else model_names
    for name in names:
        steps.append({
            "id": f"train_eval:{name}",
            "desc": f"Train + evaluate: {name}",
            "cmd": [sys.executable, "-m", "src.experiments.run_all", "--only", name],
            "artifact_ok": lambda n=name: _model_has_result(cfg, n),
            "group": "train_eval",  # see note in run_pipeline() about partial-failure handling
            "parallel_group": "train_eval",
        })

    # Independent repeats of a model config (different seed => different init, noise and batches).
    # These are what give H1/H3 two or more observations per condition.
    for name, sd in (extra_seeds or []):
        steps.append({
            "id": f"train_eval:{name}@seed{sd}",
            "desc": f"Train + evaluate: {name} (seed {sd})",
            "cmd": [sys.executable, "-m", "src.experiments.run_all", "--only", name, "--seed", str(sd)],
            "artifact_ok": lambda n=name, k=sd: _model_has_result(cfg, n, seed=k),
            "group": "train_eval",
            "parallel_group": "train_eval",
        })

    results_dir = cfg.get("logging", {}).get("results_dir", "results")
    steps.append({
        "id": "run_qpca",
        "desc": "QPCA (simulated phase estimation) on the processed features -> results/qpca_results.json (Figure 31 input)",
        "cmd": [sys.executable, "-m", "scripts.run_qpca"],
        "required": False,
        "artifact_ok": lambda: _nonempty(os.path.join(results_dir, "qpca_results.json")),
    })
    steps.append({
        "id": "make_figures",
        "always_run": True,  # reads whatever qpca / entanglement-map files exist now
        "analysis": True,
        "desc": "Draw Figure 31 (QPCA loadings) and Figure 32 (entanglement map) from saved run output",
        "cmd": [sys.executable, "-m", "scripts.make_figures"],
        "required": False,
        "artifact_ok": lambda: True,   # a missing input only skips that figure; the script exits 0
    })
    steps.append({
        "id": "verify_chapter4",
        "always_run": True,  # reads results/all_results.csv -- never reuse a stale pass
        "analysis": True,    # never blocked by, and never blocks, any other step
        "desc": "Compare actual results against dissertation claims",
        "cmd": [sys.executable, "-m", "scripts.verify_chapter4"],
        "artifact_ok": lambda: _nonempty(
            os.path.join(cfg.get("logging", {}).get("results_dir", "results"), "chapter4_numbers.json")
        ),
    })
    steps.append({
        "id": "run_hypothesis_tests",
        "always_run": True,  # depends on the latest results -- never skipped, even on "Y"
        "analysis": True,    # runs even if training or an earlier required step failed
        "desc": "Compute the actual H1-H5 statistical tests from results/all_results.csv",
        "cmd": [sys.executable, "-m", "scripts.run_hypothesis_tests"],
        "required": False,
        "artifact_ok": lambda: _nonempty(
            os.path.join(cfg.get("logging", {}).get("results_dir", "results"), "hypothesis_test_results.json")
        ),
    })
    steps.append({
        "id": "scan_logs_for_pii",
        "desc": "Data-sanitization gate: scan logs/*.json, *.log for PII patterns",
        "cmd": [sys.executable, "-m", "scripts.scan_run_artifacts_for_pii"],
        "required": False,
        "artifact_ok": lambda: True,  # a clean scan writes nothing; absence of a hit isn't an artifact to check
    })
    return steps


def _model_has_result(cfg: dict, model_name: str, seed: int | None = None) -> bool:
    results_dir = cfg.get("logging", {}).get("results_dir", "results")
    csv_path = os.path.join(results_dir, "all_results.csv")
    if not _nonempty(csv_path):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Match on the run_id slug, not model_type: every QGAN-LLM ablation
        # config is recorded as model_type "QGAN-LLM", so the old check never
        # succeeded for "(12-qubit)", "(linear topology)" or "(no noise)".
        from src.utils.run_identity import config_of_run_id, model_slug
        match = df["run_id"].map(config_of_run_id) == model_slug(model_name)
        if seed is not None:
            if "seed" not in df.columns:
                return False
            match = match & (df["seed"] == seed)
        return bool(match.any())
    except Exception:
        return False


def resolve_model_names(cfg_path: str | None, only: str | None) -> tuple[list[str], list[str] | None]:
    """Asks src/experiments/run_all.py itself (via --list-models) to
    resolve baselines + config-defined ablations into concrete names,
    rather than re-implementing that merge here and risking the two
    lists drifting apart."""
    cmd = [sys.executable, "-m", "src.experiments.run_all", "--list-models"]
    if cfg_path:
        cmd += ["--config", cfg_path]
    all_names = json.loads(subprocess.check_output(cmd, text=True))
    only_names = None
    if only:
        wanted = {n.strip() for n in only.split(",")}
        unknown = wanted - set(all_names)
        if unknown:
            raise SystemExit(f"--only names not found in config: {sorted(unknown)}\n"
                             f"Available: {all_names}")
        only_names = [n for n in all_names if n in wanted]
    return all_names, only_names


# ------------------------------------------------------------------------
# The single interactive prompt.
# ------------------------------------------------------------------------
# Step ids that must run again this invocation even though the state file
# says "completed" (filled in by the "a" answer / --rerun-failed).
_force_rerun: set = set()
_fresh_requested = False      # set by decide_skip_policy when the person chose "start from scratch"


def _mark_stale_running(state: dict) -> list:
    """A step recorded as "running" when a NEW pipeline starts belongs to a process that is gone
    (killed / crashed). Left as is it shows a run time that keeps growing (e.g. 4739m41s). Mark it
    "interrupted" and keep its start time."""
    stale = []
    with _state_lock:
        for sid, rec in state.get("steps", {}).items():
            if rec.get("status") == "running":
                rec["status"] = "interrupted"
                rec["interrupted_utc"] = _now()
                stale.append(sid)
        if stale:
            save_state(state)
    return stale


def begin_fresh_run(state: dict, steps: list, keep_llm_cache: bool = False) -> None:
    """Make "start from scratch" real. Choosing n / --no-skip / --reset used to only stop run_pipeline
    from SKIPPING steps; each step still resumed its own checkpoint, reused cached synthetic rows and
    cached LLM answers, and the dashboard kept showing the previous run's DONE/RUN lines.

    Now, for the steps in this run:
      * state records are archived and reset to "pending" (no stale DONE / RUN durations);
      * PIPELINE_FRESH=1 is exported so every step trains from epoch 0 and regenerates synthetic rows
        (see src/utils/gan_training.fresh_run); old checkpoints are overwritten and archived by the
        existing archive_file logic, never silently lost;
      * results/llm_cache/ is moved to results/history/llm_cache__<run id>/ (unless keep_llm_cache),
        so LLM answers are asked for again."""
    import shutil
    from src.utils.run_identity import archive_file, pipeline_run_id
    os.environ["PIPELINE_FRESH"] = "1"
    kept = archive_file(STATE_PATH, tag=pipeline_run_id() + "_fresh") if os.path.isfile(STATE_PATH) else None
    with _state_lock:
        for s in steps:
            state["steps"][s["id"]] = {"status": "pending"}
        state["fresh_run_utc"] = _now()
        save_state(state)
    moved = None
    cache = os.path.join("results", "llm_cache")
    if not keep_llm_cache and os.path.isdir(cache) and os.listdir(cache):
        moved = os.path.join("results", "history", f"llm_cache__{pipeline_run_id()}")
        os.makedirs(os.path.dirname(moved), exist_ok=True)
        shutil.move(cache, moved)
    print("[run_pipeline] FRESH RUN: every step below starts from scratch.\n"
          "  - step status reset to pending (previous state archived: " + str(kept) + ")\n"
          "  - checkpoints are NOT resumed; models train from epoch 0 and synthetic rows are regenerated\n"
          + (f"  - LLM answer cache moved to {moved}\n" if moved else
             "  - LLM answer cache kept (--keep-llm-cache)\n" if keep_llm_cache and os.path.isdir(cache) else ""))


def _log_shows_failure(step_id: str) -> bool:
    """True if the step's latest per-step log (logs/pipeline/<id>.log, rewritten
    on every run) contains a Python traceback. Catches steps recorded as
    "completed" whose last run actually blew up."""
    safe = step_id.replace(":", "_").replace(" ", "_").replace("/", "_")
    path = os.path.join("logs", "pipeline", f"{safe}.log")
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return "Traceback (most recent call last)" in tail


def _looks_failed(step: dict, state: dict) -> bool:
    rec = state["steps"].get(step["id"], {})
    return (rec.get("status") != "completed"
            or rec.get("returncode", 0) not in (0, None)
            or _log_shows_failure(step["id"]))


def decide_skip_policy(state: dict, steps: list[dict], args) -> bool:
    """Returns True if already-completed (and artifact-verified) steps
    should be skipped this run, False if everything should be re-run.

    Asks AT MOST ONCE, and only when there is a real decision to make
    (i.e. at least one step is both marked completed in the state file
    AND still has its artifact on disk). If nothing is skippable, there
    is nothing to ask about, so it proceeds silently -- an empty/fresh
    state file means a first run, not a question.
    """
    global _fresh_requested
    _fresh_requested = False          # decided anew on every call; never carried over from an earlier one
    if args.reset:
        if os.path.isfile(STATE_PATH):
            from src.utils.run_identity import archive_file
            kept = archive_file(STATE_PATH)           # archived, never just deleted
            os.remove(STATE_PATH)
            print(f"[run_pipeline] --reset: previous state kept at {kept}")
        print(f"[run_pipeline] --reset: removed {STATE_PATH}. Starting fresh; nothing to skip.")
        _fresh_requested = True
        return False

    skippable = [s for s in steps
                if not s.get("always_run") and not s.get("analysis")
                and state["steps"].get(s["id"], {}).get("status") == "completed" and s["artifact_ok"]()]
    if not skippable:
        return False  # first run, or state file doesn't match anything on disk -- nothing to decide

    if getattr(args, "rerun_failed", False):
        return _apply_rerun_failed(skippable, state)
    if args.yes:
        print(f"[run_pipeline] --yes: will skip {len(skippable)} previously completed step(s).")
        return True
    if args.no_skip:
        print(f"[run_pipeline] --no-skip: ignoring {len(skippable)} previously completed step(s); redoing all.")
        _fresh_requested = True
        return False

    print(f"\nFound {len(skippable)} previously completed step(s) from an earlier run of this pipeline:")
    for s in skippable:
        rec = state["steps"][s["id"]]
        flag = "   <-- log shows an error" if _looks_failed(s, state) else ""
        print(f"  - {s['id']:<28} completed {rec.get('completed_utc', '?')} "
              f"({rec.get('duration_seconds', '?')}s){flag}")

    if not sys.stdin.isatty():
        print("[run_pipeline] Non-interactive session (no TTY) and neither --yes nor --no-skip was "
              "given; defaulting to SKIP the steps above. Pass --no-skip to force a full re-run "
              "in non-interactive contexts instead.")
        return True

    answer = input("\nSkip these and continue where you left off?\n"
                   "  Y = skip them and resume unfinished work (default)\n"
                   "  n = start EVERYTHING from scratch (no checkpoints, no caches, state reset)\n"
                   "  a = re-run every step that failed, even if marked completed\n"
                   "[Y/n/a]: ").strip().lower()
    if answer in ("a", "all"):
        return _apply_rerun_failed(skippable, state)
    skip = answer in ("", "y", "yes")
    if not skip:
        _fresh_requested = True
    print(f"[run_pipeline] {'Skipping' if skip else 'Re-running'} the above for the rest of this run.\n")
    return skip


def _apply_rerun_failed(skippable: list, state: dict) -> bool:
    """Mark every 'completed' step that actually failed (non-zero return code
    or a traceback in its latest log) to run again; everything else is skipped.
    Steps whose state is failed/running are never skipped anyway."""
    bad = [s["id"] for s in skippable if _looks_failed(s, state)]
    _force_rerun.update(bad)
    if bad:
        print(f"[run_pipeline] Re-running {len(bad)} step(s) marked completed but whose last run "
              f"failed: {bad}")
    print("[run_pipeline] Failed steps will re-run; healthy completed steps are skipped.\n")
    return True


# ------------------------------------------------------------------------
# Central monitoring: one rendering function shared by (a) the live table
# shown automatically while a --jobs>1 batch runs, and (b) `--watch`, which
# attaches to an already-running pipeline from a second terminal/session.
# Both read the exact same two sources of truth: the state file (status,
# timestamps, durations) and each step's log file (a live progress hint) --
# there is no separate monitoring path to fall out of sync with reality.
# ------------------------------------------------------------------------
def _tail_last_line(path: str, max_bytes: int = 8000) -> str:
    """Returns the last rendered line of a log file, as a terminal would
    show it -- tqdm redraws a bar with '\\r', not '\\n', so a naive
    split('\\n') would return a blank last line while a bar is mid-run.
    Splitting on both catches the true most recent frame."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""
    import re
    parts = [p for p in re.split(r"[\r\n]+", chunk) if p.strip()]
    return parts[-1].strip()[:88] if parts else ""


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def render_dashboard(steps: list[dict], state: dict, log_dir: str) -> str:
    """One line per step, in pipeline order: status, elapsed/final
    duration, and -- for a currently-running step with a log file (i.e.
    one running inside a --jobs>1 batch) -- the latest line from that log,
    so a step that looks 'stuck' from the outside is visibly still making
    progress (e.g. '42% 1200/2850 batches')."""
    now = time.time()
    icons = {"completed": "[DONE]", "failed": "[FAIL]", "running": "[RUN ]", "pending": "[wait]"}
    lines = [f"\n{'=' * 78}", f" PIPELINE STATUS  {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}",
             "-" * 78]
    for s in steps:
        rec = state.get("steps", {}).get(s["id"], {})
        status = rec.get("status", "pending")
        icon = icons.get(status, "[wait]")
        if status == "running" and "started_utc" in rec:
            started = datetime.strptime(rec["started_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            dur = _fmt_duration((datetime.now(timezone.utc) - started).total_seconds())
        elif "duration_seconds" in rec:
            dur = _fmt_duration(rec["duration_seconds"])
        else:
            dur = "--"
        hint = ""
        if status == "running":
            safe = s["id"].replace(":", "_").replace(" ", "_").replace("/", "_")
            hint = _tail_last_line(os.path.join(log_dir, f"{safe}.log"))
        lines.append(f" {icon} {s['id']:<32} {dur:>7}  {hint}")
    lines.append("=" * 78)
    return "\n".join(lines)


def watch_pipeline(args) -> int:
    """Standalone monitor: attaches to a pipeline that another invocation
    of this same script (in another terminal, or another SSH session) is
    running or has run, and shows the live dashboard until every step is
    in a terminal state (completed/failed) or the person hits Ctrl-C. Does
    not run or affect anything -- purely reads the state file and log
    files that the running pipeline is already writing."""
    cfg = load_config(args.config)
    model_names, only_names = resolve_model_names(args.config, args.only)
    steps = build_steps(cfg, model_names, only_names)
    log_dir = os.path.join("logs", "pipeline")
    clear = sys.stdout.isatty()

    print(f"[run_pipeline --watch] Watching {STATE_PATH} every {args.watch_interval}s. Ctrl-C to stop.")
    try:
        while True:
            state = load_state()
            if clear:
                sys.stdout.write("\033c")  # only clear when attached to a real terminal --
                                           # never when piped/redirected to a file
            print(render_dashboard(steps, state, log_dir))
            statuses = [state.get("steps", {}).get(s["id"], {}).get("status", "pending") for s in steps]
            if statuses and all(st in ("completed", "failed") for st in statuses):
                print("\n[run_pipeline --watch] Every step has reached a final state. Exiting.")
                return 0
            time.sleep(args.watch_interval)
    except KeyboardInterrupt:
        print("\n[run_pipeline --watch] Stopped (the pipeline itself, if still running elsewhere, is unaffected).")
        return 0


# ------------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------------
_print_lock = threading.Lock()


def _write_run_record(args, steps, state, started_utc, t0, status, failed_train_eval, failed_analysis):
    """logs/pipeline/pipeline_run__<UTC stamp>.json (+ pipeline_run.json = latest): one JSON per
    run_pipeline.py call -- when it started/finished, how it ended, and every step's recorded state.
    Never raises: a logging problem must not change the pipeline's exit status."""
    try:
        from src.utils.run_identity import pipeline_run_id, save_json_timestamped, utc_stamp
        done = utc_stamp()
        st = state.get("steps", {}) if isinstance(state, dict) else {}
        rec = {
            "pipeline_run_id": pipeline_run_id(),
            "status": status,                       # complete | failed | aborted
            "started_utc": started_utc,
            "finished_utc": done,
            "duration_s": round(time.time() - t0, 1),
            "argv": sys.argv[1:],
            "config": getattr(args, "config", None),
            "failed_train_eval": list(failed_train_eval),
            "failed_analysis": list(failed_analysis),
            "steps": {s["id"]: st.get(s["id"], {"status": "pending"}) for s in steps},
            "step_logs_dir": os.path.join("logs", "pipeline"),
        }
        save_json_timestamped(os.path.join("logs", "pipeline", "pipeline_run.json"), rec, stamp=done)
    except Exception as exc:  # noqa: BLE001
        print(f"[run_pipeline] could not write the run record: {exc}")


def run_step(step: dict, state: dict, skip_completed: bool, log_to_file: str | None = None,
             env: dict | None = None) -> bool:
    """Returns True on success (including a skip), False on failure.

    log_to_file: when set (used by run_batch_parallel below), the
    subprocess's stdout/stderr go to this file instead of the terminal.
    Several subprocesses writing tqdm progress bars to the same terminal
    at once produces garbled, overlapping output -- each parallel step
    gets its own log file instead, and its path is printed so you can
    `tail -f` any one of them while the group runs.
    """
    step_id = step["id"]
    already_done = (skip_completed
                    and not step.get("always_run")
                    and not step.get("analysis")
                    and step_id not in _force_rerun
                    and state["steps"].get(step_id, {}).get("status") == "completed"
                    and step["artifact_ok"]())
    if already_done:
        rec = state["steps"][step_id]
        with _print_lock:
            print(f"{_ts()} [SKIP] {step_id} -- completed {rec.get('completed_utc', '?')} "
                  f"({rec.get('duration_seconds', '?')}s). Artifact verified present.")
        return True

    with _print_lock:
        print(f"\n{'=' * 70}\n{_ts()} [RUN ] {step_id} -- {step['desc']}\n"
              f"       $ {' '.join(step['cmd'])}"
              + (f"\n       (parallel; output -> {log_to_file})" if log_to_file else "")
              + f"\n{'=' * 70}")
    mark_running(state, step_id)
    t0 = time.time()
    if log_to_file:
        os.makedirs(os.path.dirname(log_to_file), exist_ok=True)
        with open(log_to_file, "w") as f:
            result = subprocess.run(step["cmd"], stdout=f, stderr=subprocess.STDOUT, env=env)
    else:
        result = subprocess.run(step["cmd"], env=env)
    duration = time.time() - t0

    with _print_lock:
        if result.returncode == 0:
            ok = step["artifact_ok"]()
            if not ok and step.get("required", True):
                print(f"[WARN] {step_id} exited successfully but its expected output was not found. "
                      f"Not marking it completed.")
                mark_failed(state, step_id, duration, returncode=0)
                return False
            mark_completed(state, step_id, duration,
                           detail="artifact verified" if ok else "optional step; no artifact produced")
            print(f"{_ts()} [DONE] {step_id} in {duration:.1f}s")
            return True

        mark_failed(state, step_id, duration, result.returncode)
        print(f"{_ts()} [FAIL] {step_id} exited with code {result.returncode} after {duration:.1f}s"
              + (f" -- see {log_to_file}" if log_to_file else ""))
        return False


def _child_env(n_jobs: int, step: dict) -> dict:
    """Environment for a parallel child: cap BLAS/torch threads so N jobs do
    not each grab every core (oversubscription makes parallel slower than
    sequential), and tell the child how many jobs ran at once so latency
    figures can be flagged as measured under contention."""
    env = dict(os.environ)
    if step.get("parallel_group") == "train_eval":
        threads = str(max(1, (os.cpu_count() or 1) // max(1, n_jobs)))
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(k, threads)
    env["PIPELINE_CONCURRENT_JOBS"] = str(n_jobs)
    return env


def auto_jobs(batch: list[dict]) -> int:
    """Default parallelism: acquire steps are network-bound (run them all);
    train_eval steps are CPU/RAM-bound (~2-3 GB each), so use at most half
    the cores and what RAM allows, never more than the batch size."""
    group = batch[0].get("parallel_group")
    if group == "acquire":
        return len(batch)
    cpu = os.cpu_count() or 1
    jobs = max(1, cpu // 2)
    try:
        import psutil
        jobs = min(jobs, max(1, int(psutil.virtual_memory().available / 3e9)))
    except Exception:
        pass
    return max(1, min(jobs, len(batch)))


def run_batch_parallel(batch: list[dict], state: dict, skip_completed: bool,
                       max_workers: int, log_dir: str, dashboard_interval: float = 5.0) -> dict[str, bool]:
    """Runs every step in `batch` concurrently (bounded by max_workers) and
    returns {step_id: success}. Used for the two independent groups this
    pipeline has -- acquire_* and train_eval:* (see build_steps) -- never
    across a barrier like prepare_data, which always runs alone.

    Each subprocess writes to its own logs/pipeline/<step_id>.log instead
    of the shared terminal (see run_step's log_to_file docstring), which
    would otherwise hide progress entirely once jobs>1. To make this
    launch command double as central monitoring rather than going dark
    while several steps run, a background thread redraws render_dashboard
    (the same table `--watch` shows from another terminal) every
    `dashboard_interval` seconds until the whole batch finishes -- so you
    see every task's status and live progress hint without needing a
    second terminal, `tail -f`, or the --watch flag at all; those remain
    available for attaching from elsewhere (e.g. a different SSH session).

    The shared `state` dict is mutated from multiple threads here: plain
    dict item assignment is atomic under the GIL, and save_state()'s write
    is itself atomic (temp file + os.replace, src/utils/io.py), so
    concurrent mark_*() calls are safe without an additional lock -- each
    write reflects the full, current state at that moment, just possibly
    superseded moments later by another thread's equally-valid write.
    """
    os.makedirs(log_dir, exist_ok=True)
    results: dict[str, bool] = {}
    stop_dashboard = threading.Event()

    def _dashboard_loop():
        while not stop_dashboard.wait(dashboard_interval):
            with _print_lock:
                print(render_dashboard(batch, state, log_dir))

    monitor = threading.Thread(target=_dashboard_loop, daemon=True)
    monitor.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            safe_names = {s["id"]: s["id"].replace(":", "_").replace(" ", "_").replace("/", "_") for s in batch}
            futures = {
                ex.submit(run_step, s, state, skip_completed,
                         log_to_file=os.path.join(log_dir, f"{safe_names[s['id']]}.log"),
                         env=_child_env(max_workers, s)): s
                for s in batch
            }
            for fut in concurrent.futures.as_completed(futures):
                step = futures[fut]
                results[step["id"]] = fut.result()
    finally:
        stop_dashboard.set()
        monitor.join(timeout=1.0)
    with _print_lock:
        print(render_dashboard(batch, state, log_dir))  # final snapshot, all terminal
    return results


def run_pipeline(argv=None) -> int:

    parser = argparse.ArgumentParser(description=__doc__.split("EXECUTION STEPS")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML (default: config/default_config.yaml)")
    parser.add_argument("--only", default=None,
                         help="Comma-separated subset of model names for the train_eval steps "
                              "(default: every baseline + ablation in the config)")
    parser.add_argument("--reuse-checkpoints", action="store_true",
                         help="Passed through to src.experiments.run_all for models that support it")
    parser.add_argument("--yes", action="store_true",
                         help="Non-interactive: automatically skip previously completed steps")
    parser.add_argument("--no-skip", action="store_true",
                         help="Non-interactive: redo every step, ignoring the state file")
    parser.add_argument("--rerun-failed", action="store_true",
                         help="Non-interactive equivalent of answering 'a': re-run every failed step, "
                              "including ones marked completed whose last log shows a traceback; "
                              "skip the healthy completed ones")
    parser.add_argument("--keep-llm-cache", action="store_true",
                         help="With a from-scratch run (n / --no-skip / --reset): keep results/llm_cache so "
                              "LLM answers are reused instead of re-requested (saves API time)")
    parser.add_argument("--reset", action="store_true",
                         help="Delete the state file first, then run everything (implies --no-skip)")
    parser.add_argument("--stats-only", action="store_true",
                         help="Run ONLY run_hypothesis_tests (always re-runs; needs no model listing, "
                              "no network, no NVIDIA key). Use it after a run to recompute H1-H5.")
    parser.add_argument("--analysis-only", action="store_true",
                         help="Run ONLY verify_chapter4 and run_hypothesis_tests.")
    parser.add_argument("--steps", default=None, metavar="ID[,ID...]",
                         help="Run only these step ids (e.g. run_hypothesis_tests,scan_logs_for_pii). "
                              "train_eval ids need the model name: 'train_eval:Classical GAN-LLM'.")
    parser.add_argument("--extra-seeds", type=int, default=0, metavar="N",
                         help="Also train+evaluate each --seed-models model with N additional seeds "
                              "(config seed+1 ... seed+N). Independent repeats are what H1/H3 need.")
    parser.add_argument("--seed-models", default=None, metavar="NAME[,NAME...]",
                         help="Models to repeat for --extra-seeds. Default: none (the quantum GANs "
                              "take 7-13 h per run, so you choose).")
    parser.add_argument("--status", action="store_true",
                         help="Print the current state file and exit; runs nothing")
    parser.add_argument("--watch", action="store_true",
                         help="Attach to a pipeline already running (in another terminal/session) "
                              "and show a live status table for every step until all of them "
                              "reach a final state, or Ctrl-C. Reads the same state file and "
                              "per-step logs the running pipeline writes; does not start or "
                              "affect anything itself.")
    parser.add_argument("--watch-interval", type=float, default=3.0, metavar="SECONDS",
                         help="Refresh interval for --watch (default: 3s)")
    parser.add_argument("--jobs", type=int, default=0, metavar="N",
                         help="Run independent steps within the acquire_* group and within the "
                              "train_eval:* group up to N at a time. Default 0 = AUTO (acquire: all "
                              "at once; train_eval: min(cores/2, free RAM/3GB, models)); "
                              "use --jobs 1 for fully sequential. prepare_data and "
                              "verify_chapter4 always run alone since they depend on every "
                              "member of the group before/after them. Parallel steps write to "
                              "logs/pipeline/<step_id>.log instead of the terminal, since several "
                              "progress bars writing to one terminal at once garbles the output. "
                              "Note: the quantum models (QGAN-LLM, QLSTM Forecaster, ablations) "
                              "all run on the same CPU-bound simulator, so running several of "
                              "those at once competes for the same cores rather than truly "
                              "parallelizing -- --jobs helps most for the acquire_* group and "
                              "for mixing classical + quantum training.")
    args = parser.parse_args(argv)

    if args.yes and args.no_skip:
        raise SystemExit("--yes and --no-skip are mutually exclusive (skip vs. redo everything).")

    state = load_state()

    if args.status:
        print(json.dumps(state, indent=2, default=str))
        return 0

    if args.watch:
        return watch_pipeline(args)

    cfg = load_config(args.config)

    wanted_ids = None
    if args.stats_only:
        wanted_ids = ["run_hypothesis_tests"]
    elif args.analysis_only:
        wanted_ids = ["verify_chapter4", "run_hypothesis_tests"]
    elif args.steps:
        wanted_ids = [x.strip() for x in args.steps.split(",") if x.strip()]

    extra_seeds = None
    if wanted_ids is not None and not any(w.startswith("train_eval:") for w in wanted_ids):
        # Analysis-only invocation: no need to import torch/pennylane or touch the network just to
        # list model names that will not run.
        model_names, only_names = [], None
    else:
        model_names, only_names = resolve_model_names(args.config, args.only)
        if args.extra_seeds > 0:
            chosen = [n.strip() for n in (args.seed_models or "").split(",") if n.strip()]
            unknown = [n for n in chosen if n not in model_names]
            if unknown:
                raise SystemExit(f"--seed-models names not found in config: {unknown}\n"
                                 f"Available: {model_names}")
            if not chosen:
                raise SystemExit("--extra-seeds needs --seed-models \"Model A,Model B\" "
                                 "(which models to repeat).")
            base = int(cfg.get("seed", 42))
            extra_seeds = [(n, base + k) for n in chosen for k in range(1, args.extra_seeds + 1)]
    steps = build_steps(cfg, model_names, only_names, extra_seeds)
    if wanted_ids is not None:
        known = {s["id"] for s in steps}
        missing = [w for w in wanted_ids if w not in known]
        if missing:
            raise SystemExit(f"--steps: unknown step id(s) {missing}.\nKnown: {sorted(known)}")
        steps = [s for s in steps if s["id"] in wanted_ids]
    if args.reuse_checkpoints:
        for s in steps:
            if s["id"].startswith("train_eval:"):
                s["cmd"].append("--reuse-checkpoints")

    # An explicit step selection means "run exactly these now": never skip, never prompt.
    os.environ.pop("PIPELINE_FRESH", None)   # only an explicit "from scratch" choice below may set it
    stale = _mark_stale_running(state)
    if stale:
        print(f"[run_pipeline] steps left 'running' by an earlier, dead process are now 'interrupted': {stale}")
    skip_completed = False if wanted_ids is not None else decide_skip_policy(state, steps, args)
    if _fresh_requested and wanted_ids is None:
        begin_fresh_run(state, steps, keep_llm_cache=getattr(args, "keep_llm_cache", False))

    pipeline_t0 = time.time()
    from src.utils.run_identity import utc_stamp as _utc_stamp
    started_utc = _utc_stamp()
    failed_train_eval = []
    failed_analysis = []
    log_dir = os.path.join("logs", "pipeline")

    i = 0
    while i < len(steps):
        group = steps[i].get("parallel_group")
        batch = [steps[i]]
        j = i + 1
        while group is not None and j < len(steps) and steps[j].get("parallel_group") == group:
            batch.append(steps[j])
            j += 1
        i = j

        jobs = args.jobs if args.jobs > 0 else (auto_jobs(batch) if group is not None else 1)
        if group is not None and jobs > 1 and len(batch) > 1:
            print(f"\n[run_pipeline] Running '{group}' group in parallel "
                  f"(--jobs {jobs}{' auto' if args.jobs <= 0 else ''}): {[s['id'] for s in batch]}")
            print(f"[run_pipeline] Live status will refresh below every few seconds. To watch from "
                  f"another terminal instead: python run_pipeline.py --watch"
                  + (f" --config {args.config}" if args.config else ""))
            results = run_batch_parallel(batch, state, skip_completed, jobs, log_dir)
        else:
            results = {s["id"]: run_step(s, state, skip_completed) for s in batch}

        for step in batch:
            ok = results[step["id"]]
            if ok:
                continue
            if step.get("analysis"):
                # Analysis steps are independent of each other and of training: record and go on.
                failed_analysis.append(step["id"])
                continue
            if step.get("group") == "train_eval":
                # Models are independent of each other: one failing shouldn't
                # cost you the ones that would still succeed. Collect it and
                # keep going; the run is still reported as failed overall.
                failed_train_eval.append(step["id"])
                continue
            if not step.get("required", True):
                print(f"[run_pipeline] {step['id']} is optional; continuing despite failure.")
                continue
            print(f"\n[run_pipeline] Aborting: required step '{step['id']}' failed. "
                  f"Fix the error above and re-run `python run_pipeline.py` -- "
                  f"completed steps will be skipped automatically.")
            # Still refresh the analysis from whatever results exist; it is cheap and independent.
            for a in [x for x in steps[i:] if x.get("analysis")]:
                if not run_step(a, state, False):
                    failed_analysis.append(a["id"])
            _write_run_record(args, steps, state, started_utc, pipeline_t0, "aborted",
                              failed_train_eval + [step["id"]], failed_analysis)
            return 1

    total = time.time() - pipeline_t0
    print(f"\n{'=' * 70}")
    if failed_analysis:
        print(f"ANALYSIS STEP(S) FAILED: {failed_analysis}")
        print("They re-run on every invocation; to retry only them: "
              "python run_pipeline.py --analysis-only")
    if failed_train_eval:
        print(f"PIPELINE FINISHED WITH FAILURES in {total / 60:.1f} min.")
        print(f"Failed: {failed_train_eval}")
        print(f"Re-run `python run_pipeline.py` -- everything else will be skipped "
              f"automatically and only these will retry.")
        print("=" * 70)
        _write_run_record(args, steps, state, started_utc, pipeline_t0, "failed", failed_train_eval, failed_analysis)
        return 1
    if failed_analysis:
        print("=" * 70)
        _write_run_record(args, steps, state, started_utc, pipeline_t0, "failed", failed_train_eval, failed_analysis)
        return 1

    from src.utils.run_identity import archive_file, pipeline_run_id
    archive_file(STATE_PATH, tag=pipeline_run_id())
    print(f"PIPELINE COMPLETE in {total / 60:.1f} min. State: {STATE_PATH} (copy: data/history/, run {pipeline_run_id()})")
    _write_run_record(args, steps, state, started_utc, pipeline_t0, "complete", failed_train_eval, failed_analysis)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
