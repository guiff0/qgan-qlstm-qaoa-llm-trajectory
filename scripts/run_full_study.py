r"""
Launch the study once; it stops only when every run is done.

DEFAULT = the AMENDMENT 3 plan: 4 conditions x 10 seeds = 40 runs (QGAN-LLM, Classical GAN-LLM, Classical GAN-LLM (matched),
QGAN-LLM (no noise)), fully crossed (every condition gets the SAME seeds, so groups are paired and balanced). The entanglement
sweep and its hypotheses (old H2, H5, M2) were dropped on 12 Nov 2026 to save compute: 64 -> 40 runs (80 -> 40 for --full).
The five hypotheses H1-H5 are in HYPOTHESES_V3.md. The hypothesis tests run automatically at the end. The sweep stays available
as an optional exploratory add-on: --levels 0,2,4 or --levels all (levels L0-L5, see src/evaluation/entanglement_design.py).
--quick = the old 4 conditions x 4 seeds smoke test (exploratory; the hypothesis tests are not run automatically on it). Relaunching resumes the experiment with the same plan; a plan that differs from the ACTIVE
experiment automatically starts (or resumes) its own experiment instead of refusing.

    python -m scripts.run_full_study                       # 4 conditions x 10 seeds, then the hypothesis tests
    python -m scripts.run_full_study --status              # what is done / remaining, nothing run
    python -m scripts.run_full_study --dry-run             # the plan, nothing run
    python -m scripts.run_full_study --n-runs 10 --levels all   # optional exploratory sweep
    .\scripts\run_full_study.ps1                           # same, with the venv's python (Windows)

CONDITIONS (all seeded with the same --base-seed, so groups are paired and balanced)
    QGAN-LLM                      primary quantum model                 -> H1, H2, H3, H4, H5
    Classical GAN-LLM             primary classical model               -> H1, H2, H3
    Classical GAN-LLM (matched)   parameter-matched classical bracket   -> H2, H3 (bracket)
    QGAN-LLM (no noise)           differs from the primary only in noise -> H4, H5
    QGAN-LLM (ent-L<k> ...)       OPTIONAL exploratory entanglement sweep (not in the default plan, not in H1-H5)

WHAT "ONLY STOPS WHEN DONE" MEANS
  * It repeats passes of scripts.run_repeated_experiment until every (condition, seed) has a row in
    results/all_results.csv. A failed run does not stop the others; failed runs are retried on the next
    pass after a wait (default 5 min, growing to 30).
  * Safe to kill, reboot, or re-launch at any time: finished (condition, seed) pairs are never repeated.
  * The one thing that can stop it early is a SYSTEMATIC error: if `--max-stalled-passes` passes in a row
    complete nothing new (default 5), retrying forever would only burn time, so it exits with code 1 and
    names the runs that keep failing. `--max-stalled-passes 0` = never give up.
  * `--workers N` runs N (condition, seed) runs at the same time, each in its own process with
    cpu_count // N BLAS/OpenMP threads (4 cores / 8 logical: N=2 -> 4 threads each, N=4 -> 2 each). Each run's
    console output goes to logs/study/<condition>_seed<k>.log. Two runs of the same condition never overlap. Quantum simulation is mostly single-threaded, so
    N = number of physical cores (4) usually beats N=1 with 8 threads; watch RAM (each process loads the data).
  * On Windows it asks the OS not to sleep while running (the display may still turn off).
  * When all runs are complete it runs scripts.run_hypothesis_tests and writes logs/full_study_status.json.

FIRST (unless --no-prepare) it runs `scripts.run_study --only calibrate --apply-calibration`, so the H3 poisoning strength is
calibrated and written to the config before anything else, then checks the pre-registration freeze (pass
--record-deviation "<reason>" to record an intended Amendment; otherwise unrecorded changes are listed as a WARNING).

Before starting it (a) regenerates the entanglement-sweep config block (idempotent), and (b) refuses to run if
forecast_path is not 'generator' in both GAN sections, because H4 would then be untestable
(override with --allow-head-only).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import run_repeated_experiment as rre  # noqa: E402
from src.evaluation.entanglement_design import LEVELS, sweep_name  # noqa: E402
from scripts.study_dashboard import StudyDashboard  # noqa: E402
from scripts.study_diagnose import diagnose_failure, one_line  # noqa: E402
from src.utils import experiment as exp  # noqa: E402

PRIMARY = ["QGAN-LLM", "Classical GAN-LLM", "Classical GAN-LLM (matched)", "QGAN-LLM (no noise)"]
DEFAULT_LEVELS = ()          # no entanglement sweep by default (H2/H5/M2 dropped 12 Nov 2026)
# DEFAULT PLAN = METHOD_N_RUNS (10) seeds x build_conditions(DEFAULT_LEVELS) (4 conditions) = 40 runs. `--full` = same 40.
# `--quick` = REDUCED_CONDITIONS x REDUCED_N_RUNS = the 16-run smoke test.
REDUCED_CONDITIONS = ["QGAN-LLM", "Classical GAN-LLM", "Classical GAN-LLM (matched)", "QGAN-LLM (no noise)"]
REDUCED_N_RUNS, FULL_N_RUNS, METHOD_N_RUNS = 4, 10, 10
STATUS_FILE = os.path.join("logs", "full_study_status.json")
DASH: StudyDashboard | None = None      # set in main(); None (e.g. in unit tests) = plain one-line output
EXPERIMENT: str | None = None           # experiment id of this launch (None = legacy, no id)
COMPAT_STAMP = False                    # True when the run path does not stamp experiment_id: the launcher stamps each row


def _study_dir() -> str:
    """logs/study/<experiment id>/ so two experiments on one machine never share a run log."""
    return os.path.join("logs", "study", EXPERIMENT) if EXPERIMENT else os.path.join("logs", "study")


def _failure_log() -> str:
    return os.path.join(_study_dir(), "failures.log")


def _log_path(model: str, seed: int) -> str:
    return os.path.join(_study_dir(), f"{rre._model_slug(model)}_seed{seed}.log")


def resolve_experiment(models, seeds, base_seed: int, config: str | None, *, experiment_id=None, new=False,
                       label: str = "", legacy: bool = False, create: bool = True, csv_path: str | None = None,
                       root: str = exp.EXPERIMENT_ROOT, parent: str | None = None, auto_adopt: bool = True) -> str | None:
    """Decide which experiment this launch belongs to. Never guesses when a mistake would mix two studies.

      --legacy              no id (old behaviour; rows are 'unassigned')
      --experiment-id X     join/resume X (created with this plan if new; a different plan under the same id is refused)
      --new-experiment      a fresh id (machine name + date + random suffix)
      neither               resume the ACTIVE experiment when its plan matches; start a new one on a clean results
                            folder; when results already hold unlabelled rows, create a DEFAULT experiment and adopt the
                            rows that belong to this plan (same conditions and seeds), so finished work is not repeated.
                            Rows outside the plan stay 'unassigned'. --no-auto-adopt refuses instead."""
    if legacy:
        return None
    csv_path = csv_path or rre.RESULTS_CSV
    if experiment_id:
        eid = experiment_id
    elif new:
        eid = exp.new_experiment_id()
    else:
        active = exp.get_active(root)
        if active:
            man = exp.load_manifest(active, root)
            if sorted(man["conditions"]) != sorted(models) or sorted(man["seeds"]) != sorted(seeds):
                if not create:                      # --status / --dry-run only look; they never refuse
                    return None
                raise SystemExit(
                    f"[experiment] the ACTIVE experiment {active} has a different plan "
                    f"({len(man['conditions'])} conditions, seeds {min(man['seeds'])}-{max(man['seeds'])}). "
                    f"Pass --new-experiment to start another study, or --experiment-id to name one.")
            eid = active
        else:
            present = {}
            if os.path.isfile(csv_path):
                import pandas as pd
                present = exp.experiments_in(pd.read_csv(csv_path, usecols=lambda c: c in ("run_id", "experiment_id")))
            if present:
                listing = ", ".join(f"{k} ({v} rows)" for k, v in sorted(present.items()))
                if not create:                      # --status / --dry-run only look; they never refuse
                    print(f"[experiment] note: {csv_path} holds rows ({listing}) and no experiment is active; "
                          f"showing the legacy view. See `python -m scripts.experiments list`.")
                    return None
                if not auto_adopt:
                    raise SystemExit(
                        f"[experiment] {csv_path} already holds rows ({listing}) but no experiment is active, so this "
                        f"launch cannot tell whether it continues them. Choose one:\n"
                        f"  continue earlier unlabelled runs : python -m scripts.experiments adopt --since <UTC date> --apply\n"
                        f"                                      then re-launch (it resumes the adopted experiment)\n"
                        f"  start a separate study           : add --new-experiment\n"
                        f"  old behaviour, no id             : add --legacy")
                from scripts.experiments import adopt
                eid = exp.new_experiment_id()
                rep = adopt(eid, "", conditions=models, seeds=seeds, base_seed=base_seed,
                            label=label or "default experiment (adopted earlier unlabelled rows of this plan)",
                            csv_path=csv_path, root=root, apply=True, config_path=config)
                print(f"[experiment] no experiment was active and {csv_path} holds unlabelled rows ({listing}). "
                      f"Created the DEFAULT experiment {eid} and adopted {rep['rows_to_stamp']} row(s) that match this plan "
                      f"({', '.join(f'{k}: {v}' for k, v in sorted(rep['per_condition'].items())) or 'none'}); "
                      f"{rep['unlabelled_rows_left_alone']} other unlabelled row(s) stay 'unassigned'. "
                      f"The CSV was archived first ({rep.get('csv_archive', 'n/a')}).", flush=True)
            else:
                eid = exp.new_experiment_id()
    if not create:
        return eid
    exp.create_experiment(eid, conditions=models, seeds=seeds, base_seed=base_seed, label=label,
                          config_path=config, root=root, parent=parent)
    exp.set_active(eid, root)
    return eid


def build_conditions(levels) -> list[str]:
    by_level = {lv: (topo, depth) for lv, topo, depth in LEVELS}
    if levels == "all":
        levels = sorted(by_level)
    unknown = [lv for lv in levels if lv not in by_level]
    if unknown:
        raise SystemExit(f"unknown sweep level(s) {unknown}; available: {sorted(by_level)}")
    return PRIMARY + [sweep_name(lv, *by_level[lv]) for lv in levels]


def parse_levels(text: str):
    if text.strip().lower() == "all":
        return "all"
    return tuple(int(x) for x in text.split(",") if x.strip() != "")


def done_seeds(model: str) -> set[int]:
    return set(rre._load_state(rre._default_tag(model))["completed_seeds"]) | rre._csv_seeds(model)


def remaining(models, seeds) -> list[tuple[str, int]]:
    """(model, seed) pairs still missing, seed-major so an early stop leaves balanced groups."""
    return [(m, s) for s in seeds for m in models if s not in done_seeds(m)]


def worker_env(workers: int, logical_cpus: int | None = None) -> dict:
    """Environment for one worker: the machine's logical CPUs split evenly, at least 1 thread each."""
    threads = str(max(1, (logical_cpus or os.cpu_count() or 1) // max(1, workers)))
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = threads
    return env


def _mark_state(model: str, seed: int, ok: bool, lock) -> None:
    with lock:
        tag = rre._default_tag(model)
        st = rre._load_state(tag)
        if ok:
            if seed not in st["completed_seeds"]:
                st["completed_seeds"].append(seed)
            if seed in st["failed_seeds"]:
                st["failed_seeds"].remove(seed)
        elif seed not in st["failed_seeds"]:
            st["failed_seeds"].append(seed)
        rre._save_state(tag, st)


def _run_one(model: str, seed: int, config, env: dict, lock) -> bool:
    """One (condition, seed) run in its own process; output to logs/study/. True only if its row was recorded."""
    os.makedirs(_study_dir(), exist_ok=True)         # logs/study/<experiment id>/ (created on first use)
    log = _log_path(model, seed)
    cmd = [sys.executable, "-m", "src.experiments.run_all", "--only", model, "--seed", str(seed)]
    if config:
        cmd += ["--config", config]
    if DASH:
        DASH.run_started(model, seed, log)
    else:
        print(f"[worker] start  {model!r} seed {seed}  (log: {log})", flush=True)
    t0 = time.time()
    t0_utc = datetime.now(timezone.utc).isoformat()
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n===== attempt started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if COMPAT_STAMP and EXPERIMENT and rc == 0:          # run path without stamping: label the row it just wrote
        try:
            exp.stamp_new_rows(rre.RESULTS_CSV, EXPERIMENT, rre._model_slug(model), seed, t0_utc, machine=exp.machine_slug())
        except Exception as exc:  # noqa: BLE001
            print(f"[experiment] could not stamp the row of {model!r} seed {seed}: {exc!r}", flush=True)
    recorded = seed in rre._csv_seeds(model)
    ok = rc == 0 and recorded
    _mark_state(model, seed, ok, lock)
    minutes = (time.time() - t0) / 60
    why = ""
    if not ok:
        try:
            why = one_line(diagnose_failure(log, rc, recorded, minutes))
        except Exception as exc:  # noqa: BLE001  -- the diagnosis must never break the study
            why = f"why: could not classify the failure ({exc!r}) -> fix: read {log}"
        with lock:
            os.makedirs(_study_dir(), exist_ok=True)
            with open(_failure_log(), "a", encoding="utf-8") as ff:
                ff.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {model} seed {seed} exit {rc} "
                         f"after {minutes:.1f} min  {why}\n")
    if EXPERIMENT:
        try:
            exp.register_run(EXPERIMENT, {
                "condition": model, "slug": rre._model_slug(model), "seed": seed, "status": "ok" if ok else "failed",
                "exit_code": rc, "minutes": round(minutes, 2), "log": log,
                "run_id": exp.find_run_id(rre.RESULTS_CSV, EXPERIMENT, rre._model_slug(model), seed) if ok else None,
                "pipeline_run_id": env.get("PIPELINE_RUN_ID"), "machine": exp.machine_slug(),
                "why": why or None})
        except Exception as exc:  # noqa: BLE001  -- bookkeeping must never stop a multi-day study
            print(f"[experiment] could not register run: {exc!r}", flush=True)
    if DASH:
        DASH.run_finished(model, seed, ok, rc, log, why)
    else:
        print(f"[worker] {'OK   ' if ok else 'FAIL '} {model!r} seed {seed} exit {rc} in {minutes:.1f} min"
              + (f"\n         {why}" if why else ""), flush=True)
    return ok


def _run_pass_parallel(models, n_runs, base_seed, config, workers: int) -> int:
    """One pass over everything missing with `workers` runs at a time. Queue order is seed-major, so an
    interruption leaves balanced groups. Returns 0 if every attempted run succeeded, else 1."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    seeds = [base_seed + i for i in range(n_runs)]
    todo = remaining(models, seeds)
    env, lock = worker_env(workers), threading.Lock()
    print(f"[workers] {len(todo)} runs, {workers} at a time, "
          f"{env['OMP_NUM_THREADS']} thread(s) each", flush=True)
    # Never two runs of the SAME condition at once: a run_id is "<model slug>_<UTC second>" and is claimed
    # check-then-create (run_identity.unique_run_id), so two same-model processes starting in the same second
    # could share a run_id and overwrite each other's log. Different conditions have different slugs.
    from concurrent.futures import FIRST_COMPLETED, wait
    pending, active, results = list(todo), {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while pending or active:
            i = 0
            while len(active) < workers and i < len(pending):
                model, seed = pending[i]
                if model in active.values():
                    i += 1                              # that condition is already running; take the next
                    continue
                pending.pop(i)
                active[ex.submit(_run_one, model, seed, config, env, lock)] = model
            done, _ = wait(list(active), return_when=FIRST_COMPLETED)
            for fut in done:
                results.append(fut.result())
                del active[fut]
    return 0 if all(results) else 1


def _run_pass(models, n_runs, base_seed, config) -> int:
    """One pass over everything still missing. max-consecutive-failures=0: one failing run must not block the rest."""
    cmd = [sys.executable, "-m", "scripts.run_repeated_experiment", "--model", ",".join(models),
           "--n-runs", str(n_runs), "--base-seed", str(base_seed), "--max-consecutive-failures", "0"]
    if config:
        cmd += ["--config", config]
    if EXPERIMENT:
        cmd += ["--experiment-id", EXPERIMENT]
    return subprocess.run(cmd).returncode


def apply_sweep_config(config_path: str) -> None:
    r = subprocess.run([sys.executable, "-m", "scripts.gen_entanglement_sweep", "--apply", "--config", config_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"could not write the sweep config block:\n{r.stdout}\n{r.stderr}")
    bak = config_path + ".bak"
    if os.path.isfile(bak):          # the generator keeps one backup; it is clutter for a repo
        try:
            os.remove(bak)
        except OSError:
            pass


def check_forecast_path(config_path: str) -> list[str]:
    import yaml
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [f"{sec}.forecast_path = {cfg.get(sec, {}).get('forecast_path')!r}"
            for sec in ("qgan_llm", "classical_gan_llm") if cfg.get(sec, {}).get("forecast_path") != "generator"]


def keep_awake() -> None:
    """Windows: ES_CONTINUOUS | ES_SYSTEM_REQUIRED for this process. Best effort, silent elsewhere."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:  # noqa: BLE001
        pass


def write_status(models, seeds, state: str, extra: dict | None = None) -> dict:
    todo = remaining(models, seeds)
    st = {"state": state, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "models": models, "seeds": seeds, "total": len(models) * len(seeds),
          "completed": len(models) * len(seeds) - len(todo),
          "remaining": [{"model": m, "seed": s} for m, s in todo]}
    st.update(extra or {})
    st["experiment_id"] = EXPERIMENT
    targets = [STATUS_FILE] + ([os.path.join("logs", "experiments", EXPERIMENT, "full_study_status.json")]
                               if EXPERIMENT else [])
    for target in targets:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f, indent=2)
        os.replace(tmp, target)
    return st


def run_study(models, n_runs, base_seed, config, max_stalled: int, wait_s: float, sleep=time.sleep,
              run_pass=None) -> int:
    """Passes until nothing is missing. Returns 0 when complete, 1 when it gave up on a stalled study."""
    run_pass = run_pass or _run_pass
    seeds = [base_seed + i for i in range(n_runs)]
    stalled, passes, t_start = 0, 0, time.time()
    n_start = len(remaining(models, seeds))
    while True:
        todo = remaining(models, seeds)
        if not todo:
            write_status(models, seeds, "complete")
            return 0
        passes += 1
        elapsed_h = (time.time() - t_start) / 3600
        print(f"\n########## pass {passes}: {len(todo)} of {len(models) * n_runs} runs remaining "
              f"({elapsed_h:.1f} h elapsed) ##########", flush=True)
        write_status(models, seeds, "running", {"pass": passes})
        rc = run_pass(models, n_runs, base_seed, config)
        after = remaining(models, seeds)
        progressed = len(after) < len(todo)
        done_now = n_start - len(after)
        if after and done_now > 0:
            eta = (time.time() - t_start) / done_now * len(after) / 3600
            print(f"[study] {len(after)} remaining; ~{eta:.1f} h left at this session's pace", flush=True)
        if not after:
            continue                                   # loop head records 'complete'
        stalled = 0 if progressed else stalled + 1
        if max_stalled and stalled >= max_stalled:
            bad = ", ".join(f"{m} seed {s}" for m, s in after[:8])
            write_status(models, seeds, "stalled", {"pass": passes, "last_exit_code": rc})
            print(f"\n[study] STOPPED: {stalled} passes in a row completed nothing new (exit code {rc}). "
                  f"Still missing: {bad}{' ...' if len(after) > 8 else ''}. "
                  f"Look at the error above/in logs/, fix it, and re-launch the same command (it resumes). "
                  f"Use --max-stalled-passes 0 to retry forever.", flush=True)
            return 1
        pause = min(wait_s * (2 ** max(stalled - 1, 0)), 1800) if not progressed else wait_s
        print(f"[study] {'retrying' if not progressed else 'next pass'} in {pause:.0f}s "
              f"({'no progress' if not progressed else 'some runs failed and will be retried'})", flush=True)
        sleep(pause)


def default_plan_experiment(models, seeds) -> tuple[str | None, bool]:
    """(experiment id to resume, start_new) when the user named no experiment. Resumes the ACTIVE one if its plan matches,
    else the newest experiment whose plan matches, else asks for a new one when the active plan differs."""
    try:
        active = exp.get_active()
        mans = exp.list_experiments()
    except Exception:  # noqa: BLE001
        return None, False
    same = [m for m in mans if sorted(m["conditions"]) == sorted(models) and sorted(m["seeds"]) == sorted(seeds)]
    if active and any(m["experiment_id"] == active for m in same):
        return active, False
    if same:
        return same[-1]["experiment_id"], False
    return None, bool(active)


def prepare_study(args) -> None:
    """Calibrate H3 (scripts.run_study --only calibrate --apply-calibration), then check the pre-registration freeze.
    Calibration failing stops the launch; an unrecorded freeze drift only WARNS here (the strict gate is
    `scripts.run_study`, stage freeze) unless --strict-freeze is given. Idempotent: no-op when already calibrated."""
    from scripts import run_study as rs
    print("\n[prepare] 1/2 calibrate (python -m scripts.run_study --only calibrate --apply-calibration)", flush=True)
    argv = ["--only", "calibrate", "--force", "--config", args.config]
    argv += ["--allow-uncalibrated"] if args.allow_uncalibrated else ["--apply-calibration"]
    rc = rs.main(argv)
    if rc != 0:
        raise SystemExit(f"[prepare] calibration stopped the launch (rc={rc}); see the message above.")
    print("\n[prepare] 2/2 pre-registration freeze check", flush=True)
    rc = subprocess.run([sys.executable, "-m", "scripts.freeze_preregistration", "--check"]).returncode
    if rc == 0:
        print("[prepare] freeze: clean", flush=True)
        return
    if args.record_deviation:
        rc = rs.main(["--only", "freeze", "--force", "--config", args.config, "--record-deviation", args.record_deviation])
        if rc != 0:
            raise SystemExit(f"[prepare] recording the deviation failed (rc={rc}).")
        print("[prepare] freeze: deviation recorded", flush=True)
        return
    msg = ("[prepare] WARNING: tracked files differ from the pre-registration freeze (listed above). If this is the intended "
           "Amendment, record it once: .\\scripts\\run_full_study.ps1 --record-deviation \"<reason>\" ... "
           "(margins, alpha and seeds cannot be recorded away).")
    if args.strict_freeze:
        raise SystemExit(msg.replace("WARNING", "STOP"))
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-runs", type=int, default=None,
                    help="seeds per condition (default 8 = methodology plan; 10 with --full, as pre-registered; 4 with --quick)")
    ap.add_argument("--full", action="store_true",
                    help="the pre-registered plan: 8 conditions x 10 seeds = 80 runs (default: 8 conditions x 8 seeds = 64)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 4 conditions x 4 seeds = 16 runs (exploratory; H2, H5, M2 cannot be decided)")
    ap.add_argument("--hypothesis-tests", action="store_true",
                    help="run the hypothesis tests when all runs finish (default: only with --full)")
    ap.add_argument("--base-seed", type=int, default=43)
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)),
                    help="OPTIONAL exploratory entanglement-sweep levels, e.g. 0,2,5 or 'all' (default: none)")
    ap.add_argument("--config", default="config/default_config.yaml")
    ap.add_argument("--workers", type=int, default=1,
                    help="runs executed at the same time (default 1); threads per run = logical CPUs // workers. "
                         "4 cores / 8 logical: try 2-4")
    ap.add_argument("--max-stalled-passes", type=int, default=5,
                    help="give up after this many passes in a row with no new completed run (0 = never)")
    ap.add_argument("--retry-wait", type=float, default=300.0, help="seconds between passes (default 300)")
    ap.add_argument("--allow-head-only", action="store_true", help="skip the forecast_path=generator check")
    ap.add_argument("--no-hypothesis-tests", action="store_true")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="plain output; with --workers 1 this uses the old sequential pass")
    ap.add_argument("--status-interval", type=float, default=120.0,
                    help="seconds between the live per-run subtask blocks (default 120)")
    ap.add_argument("--experiment-id", default=None, help="join or resume this experiment (EXP-YYYYMMDD-MACHINE-xxxx)")
    ap.add_argument("--new-experiment", action="store_true",
                    help="start a NEW experiment (use on a second machine, or for a second 80-run study)")
    ap.add_argument("--label", default="", help="free text stored in the experiment manifest")
    ap.add_argument("--legacy", action="store_true", help="no experiment id (old behaviour)")
    ap.add_argument("--no-auto-adopt", action="store_true",
                    help="with unlabelled rows and no active experiment, refuse instead of creating a default experiment")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated condition names that REPLACE the default plan (used by the sensitivity study)")
    ap.add_argument("--parent-experiment", default=None,
                    help="experiment this one is a sensitivity/follow-up of (stored in the manifest)")
    ap.add_argument("--skip-balance-check", action="store_true",
                    help="skip the quantum-vs-classical balance audit before launching")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="run H3 with the provisional poisoning.shift_sd (rows are flagged poisoning_calibrated=False)")
    ap.add_argument("--no-prepare", action="store_true",
                    help="skip the calibrate + freeze-check step that otherwise runs first (scripts.run_study uses this: it already ran them)")
    ap.add_argument("--record-deviation", default=None, metavar="REASON",
                    help="record the pre-registration Amendment once, in the prepare step")
    ap.add_argument("--strict-freeze", action="store_true",
                    help="stop (instead of warning) when tracked files differ from the pre-registration freeze")
    ap.add_argument("--apply-calibration", action="store_true",
                    help="accepted for convenience; the prepare step always applies the calibration it computes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    global EXPERIMENT, COMPAT_STAMP

    if not (args.status or args.dry_run or args.no_prepare):
        prepare_study(args)       # calibration FIRST: it edits the config, which the experiment manifest hashes
    if args.n_runs is None:
        args.n_runs = FULL_N_RUNS if args.full else (REDUCED_N_RUNS if args.quick else METHOD_N_RUNS)
    if args.conditions:
        models = [m.strip() for m in args.conditions.split(",") if m.strip()]
    elif args.quick and not args.full:
        models = list(REDUCED_CONDITIONS)
    else:
        models = build_conditions(parse_levels(args.levels))
    reduced = args.quick and not args.full and not args.conditions
    if reduced:
        print(f"[plan] REDUCED plan: {len(models)} conditions x {args.n_runs} seeds = {len(models) * args.n_runs} runs "
              f"(not the pre-registered 80; use --full for that)", flush=True)
        if not args.label:
            args.label = f"Reduced {len(models) * args.n_runs}-run plan ({len(models)} conditions x {args.n_runs} seeds)"
    seeds = [args.base_seed + i for i in range(args.n_runs)]
    read_only = args.status or args.dry_run
    if args.experiment_id and not exp.is_valid_id(args.experiment_id):
        raise SystemExit(f"--experiment-id {args.experiment_id!r} is not of the form EXP-YYYYMMDD-MACHINE-xxxx")
    if not args.legacy and not read_only and not exp.supports_stamping():
        COMPAT_STAMP = True
        print("[experiment] src/utils/logging_utils.py does not write experiment_id yet (phase B of APPLY.md is not "
              "installed). Not a problem: the launcher stamps each finished run's results row with the experiment id "
              "itself. Predictions and compute times are only recorded once phase B is in.", flush=True)
    if not (args.experiment_id or args.new_experiment or args.legacy):
        found, start_new = default_plan_experiment(models, seeds)
        if found:
            args.experiment_id = found
        elif start_new:
            args.new_experiment = True
    EXPERIMENT = resolve_experiment(models, seeds, args.base_seed, args.config, experiment_id=args.experiment_id,
                                    new=args.new_experiment, label=args.label, legacy=args.legacy,
                                    create=not read_only, parent=args.parent_experiment,
                                    auto_adopt=not args.no_auto_adopt)
    rre.set_experiment_id(EXPERIMENT)
    print(f"[experiment] {EXPERIMENT or 'legacy (no experiment id)'}  machine={exp.machine_slug()}")
    todo = remaining(models, seeds)
    print(f"[study] {len(models)} conditions x {args.n_runs} seeds = {len(models) * args.n_runs} runs; "
          f"{len(todo)} remaining")
    for m in models:
        d = done_seeds(m) & set(seeds)
        print(f"   {m:<34} {len(d)}/{len(seeds)}")
    if args.status or args.dry_run:
        if args.dry_run:
            for m, s in todo:
                print(f"   would run: {m!r} seed={s}")
        return

    from scripts.run_study import needs_calibration          # stdlib-only helper: H3 strength must be calibrated first
    with open(args.config, encoding="utf8", newline="") as _fh:
        if needs_calibration(_fh.read()) and not args.allow_uncalibrated:
            raise SystemExit("[study] refusing to start: poisoning.protocol is v2 but poisoning.calibrated is false. Run "
                             "`python -m scripts.run_study --apply-calibration` (or scripts.calibrate_poisoning and paste the value), "
                             "or pass --allow-uncalibrated to run with the provisional strength.")
    if not args.allow_head_only:
        bad = check_forecast_path(args.config)
        if bad:
            raise SystemExit("[study] refusing to start: " + "; ".join(bad) +
                             ". H4 is untestable unless the generator is on the forecast path in BOTH GAN sections "
                             "(set forecast_path: generator, or pass --allow-head-only).")
    apply_sweep_config(args.config)
    if not args.skip_balance_check:                      # reviewer claim 2: refuse to launch an unbalanced comparison
        from scripts.check_balance import run as balance_run
        from src.evaluation.balance_audit import format_report
        rep = balance_run(args.config, EXPERIMENT)
        print(format_report(rep), flush=True)
        if not rep["pass"]:
            raise SystemExit("[study] refusing to start: the quantum-vs-classical comparison is not balanced "
                             "(see the failed checks above; --skip-balance-check overrides)")
    known = rre._known_models(args.config)       # a misspelled/missing condition would otherwise fail every pass
    if known is not None and [m for m in models if m not in known]:
        raise SystemExit(f"[study] unknown model name(s) {[m for m in models if m not in known]}; run_all knows {known}")
    keep_awake()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    pass_fn = None
    if args.workers > 1 or not args.no_dashboard or COMPAT_STAMP:   # one worker still goes through this path
        w = args.workers
        pass_fn = lambda ms, n, b, c: _run_pass_parallel(ms, n, b, c, w)   # noqa: E731
    if not args.no_dashboard:
        global DASH
        DASH = StudyDashboard(models, seeds, lambda: remaining(models, seeds), _log_path,
                              interval=args.status_interval, experiment_id=EXPERIMENT)
        DASH.header()
        DASH.start_monitor()
    rc = run_study(models, args.n_runs, args.base_seed, args.config, args.max_stalled_passes, args.retry_wait,
                   run_pass=pass_fn)
    if rc != 0:
        sys.exit(rc)
    print("\n[study] ALL RUNS COMPLETE.", flush=True)
    if reduced and not args.hypothesis_tests:
        print("[study] reduced plan: hypothesis tests not run automatically; add --hypothesis-tests (or run them yourself).", flush=True)
    if not args.no_hypothesis_tests and (args.full or not args.quick or args.conditions or args.hypothesis_tests):
        cmd = [sys.executable, "-m", "scripts.run_hypothesis_tests"]
        if EXPERIMENT:
            cmd += ["--experiment-id", EXPERIMENT]
        rc = subprocess.run(cmd).returncode
        if rc == 3:       # scripts.run_hypothesis_tests GATE_RC: deliberate stop at the confirmatory gate, not a failure
            print("\n[study] STOPPED AT THE CONFIRMATORY GATE (by design). Do the gate by hand (steps printed above), "
                  "then: python -m scripts.run_hypothesis_tests --run-confirmatory", flush=True)
            sys.exit(0)
        sys.exit(rc)


if __name__ == "__main__":
    main()
