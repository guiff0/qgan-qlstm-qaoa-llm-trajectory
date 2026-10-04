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
================================================================================
"""
from __future__ import annotations

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
    state["updated_utc"] = _now()
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    atomic_write_text(STATE_PATH, json.dumps(state, indent=2, default=str))


def mark_running(state: dict, step_id: str) -> None:
    state["steps"][step_id] = {"status": "running", "started_utc": _now()}
    save_state(state)


def mark_completed(state: dict, step_id: str, duration_s: float, detail: str = "") -> None:
    state["steps"][step_id] = {
        "status": "completed",
        "completed_utc": _now(),
        "duration_seconds": round(duration_s, 1),
        "detail": detail,
    }
    save_state(state)


def mark_failed(state: dict, step_id: str, duration_s: float, returncode: int) -> None:
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


def build_steps(cfg: dict, model_names: list[str], only_models: list[str] | None) -> list[dict]:
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

    steps.append({
        "id": "verify_chapter4",
        "desc": "Compare actual results against dissertation claims",
        "cmd": [sys.executable, "-m", "scripts.verify_chapter4"],
        "artifact_ok": lambda: _nonempty(
            os.path.join(cfg.get("logging", {}).get("results_dir", "results"), "chapter4_numbers.json")
        ),
    })
    steps.append({
        "id": "run_hypothesis_tests",
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


def _model_has_result(cfg: dict, model_name: str) -> bool:
    results_dir = cfg.get("logging", {}).get("results_dir", "results")
    csv_path = os.path.join(results_dir, "all_results.csv")
    if not _nonempty(csv_path):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        return bool((df.get("model_type") == model_name).any())
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
def decide_skip_policy(state: dict, steps: list[dict], args) -> bool:
    """Returns True if already-completed (and artifact-verified) steps
    should be skipped this run, False if everything should be re-run.

    Asks AT MOST ONCE, and only when there is a real decision to make
    (i.e. at least one step is both marked completed in the state file
    AND still has its artifact on disk). If nothing is skippable, there
    is nothing to ask about, so it proceeds silently -- an empty/fresh
    state file means a first run, not a question.
    """
    if args.reset:
        if os.path.isfile(STATE_PATH):
            os.remove(STATE_PATH)
        print(f"[run_pipeline] --reset: removed {STATE_PATH}. Starting fresh; nothing to skip.")
        return False

    skippable = [s for s in steps
                if state["steps"].get(s["id"], {}).get("status") == "completed" and s["artifact_ok"]()]
    if not skippable:
        return False  # first run, or state file doesn't match anything on disk -- nothing to decide

    if args.yes:
        print(f"[run_pipeline] --yes: will skip {len(skippable)} previously completed step(s).")
        return True
    if args.no_skip:
        print(f"[run_pipeline] --no-skip: ignoring {len(skippable)} previously completed step(s); redoing all.")
        return False

    print(f"\nFound {len(skippable)} previously completed step(s) from an earlier run of this pipeline:")
    for s in skippable:
        rec = state["steps"][s["id"]]
        print(f"  - {s['id']:<28} completed {rec.get('completed_utc', '?')} "
              f"({rec.get('duration_seconds', '?')}s)")

    if not sys.stdin.isatty():
        print("[run_pipeline] Non-interactive session (no TTY) and neither --yes nor --no-skip was "
              "given; defaulting to SKIP the steps above. Pass --no-skip to force a full re-run "
              "in non-interactive contexts instead.")
        return True

    answer = input("\nSkip these and continue where you left off? [Y/n]: ").strip().lower()
    skip = answer in ("", "y", "yes")
    print(f"[run_pipeline] {'Skipping' if skip else 'Re-running'} the above for the rest of this run.\n")
    return skip


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


def run_step(step: dict, state: dict, skip_completed: bool, log_to_file: str | None = None) -> bool:
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
                    and state["steps"].get(step_id, {}).get("status") == "completed"
                    and step["artifact_ok"]())
    if already_done:
        rec = state["steps"][step_id]
        with _print_lock:
            print(f"[SKIP] {step_id} -- completed {rec.get('completed_utc', '?')} "
                  f"({rec.get('duration_seconds', '?')}s). Artifact verified present.")
        return True

    with _print_lock:
        print(f"\n{'=' * 70}\n[RUN ] {step_id} -- {step['desc']}\n"
              f"       $ {' '.join(step['cmd'])}"
              + (f"\n       (parallel; output -> {log_to_file})" if log_to_file else "")
              + f"\n{'=' * 70}")
    mark_running(state, step_id)
    t0 = time.time()
    if log_to_file:
        os.makedirs(os.path.dirname(log_to_file), exist_ok=True)
        with open(log_to_file, "w") as f:
            result = subprocess.run(step["cmd"], stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(step["cmd"])
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
            print(f"[DONE] {step_id} in {duration:.1f}s")
            return True

        mark_failed(state, step_id, duration, result.returncode)
        print(f"[FAIL] {step_id} exited with code {result.returncode} after {duration:.1f}s"
              + (f" -- see {log_to_file}" if log_to_file else ""))
        return False


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
                         log_to_file=os.path.join(log_dir, f"{safe_names[s['id']]}.log")): s
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
    parser.add_argument("--reset", action="store_true",
                         help="Delete the state file first, then run everything (implies --no-skip)")
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
    parser.add_argument("--jobs", type=int, default=1, metavar="N",
                         help="Run independent steps within the acquire_* group and within the "
                              "train_eval:* group up to N at a time (default: 1 = fully "
                              "sequential, identical to earlier behavior). prepare_data and "
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
    model_names, only_names = resolve_model_names(args.config, args.only)
    steps = build_steps(cfg, model_names, only_names)
    if args.reuse_checkpoints:
        for s in steps:
            if s["id"].startswith("train_eval:"):
                s["cmd"].append("--reuse-checkpoints")

    skip_completed = decide_skip_policy(state, steps, args)

    pipeline_t0 = time.time()
    failed_train_eval = []
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

        if group is not None and args.jobs > 1 and len(batch) > 1:
            print(f"\n[run_pipeline] Running '{group}' group in parallel "
                  f"(--jobs {args.jobs}): {[s['id'] for s in batch]}")
            print(f"[run_pipeline] Live status will refresh below every few seconds. To watch from "
                  f"another terminal instead: python run_pipeline.py --watch"
                  + (f" --config {args.config}" if args.config else ""))
            results = run_batch_parallel(batch, state, skip_completed, args.jobs, log_dir)
        else:
            results = {s["id"]: run_step(s, state, skip_completed) for s in batch}

        for step in batch:
            ok = results[step["id"]]
            if ok:
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
            return 1

    total = time.time() - pipeline_t0
    print(f"\n{'=' * 70}")
    if failed_train_eval:
        print(f"PIPELINE FINISHED WITH FAILURES in {total / 60:.1f} min.")
        print(f"Failed: {failed_train_eval}")
        print(f"Re-run `python run_pipeline.py` -- everything else will be skipped "
              f"automatically and only these will retry.")
        print("=" * 70)
        return 1

    print(f"PIPELINE COMPLETE in {total / 60:.1f} min. State: {STATE_PATH}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
