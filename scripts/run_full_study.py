r"""
Launch the WHOLE confirmatory study once; it stops only when every run is done.

    python -m scripts.run_full_study                       # 8 conditions x 10 seeds, then the hypothesis tests
    python -m scripts.run_full_study --status              # what is done / remaining, nothing run
    python -m scripts.run_full_study --dry-run             # the plan, nothing run
    python -m scripts.run_full_study --n-runs 8 --levels all
    .\scripts\run_full_study.ps1                           # same, with the venv's python (Windows)

CONDITIONS (all seeded with the same --base-seed, so groups are paired and balanced)
    QGAN-LLM                      primary quantum model                 -> H1, H4
    Classical GAN-LLM             primary classical model               -> H1, H4
    QGAN-LLM (no noise)           differs from the primary only in noise -> H3
    QGAN-LLM (ent-L<k> ...)       controlled entanglement sweep levels   -> H2, H5, M2
                                  (default levels 0,2,4,6: none / linear x2 / ring x4 / full x4)

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

Before starting it (a) regenerates the entanglement-sweep config block (idempotent), and (b) refuses to run if
forecast_path is not 'generator' in both GAN sections, because H4 would then be untestable
(override with --allow-head-only).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import run_repeated_experiment as rre  # noqa: E402
from src.evaluation.entanglement_design import LEVELS, sweep_name  # noqa: E402

PRIMARY = ["QGAN-LLM", "Classical GAN-LLM", "Classical GAN-LLM (matched)", "QGAN-LLM (no noise)"]
DEFAULT_LEVELS = (0, 2, 4, 6)
STATUS_FILE = os.path.join("logs", "full_study_status.json")


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
    log_dir = os.path.join("logs", "study")
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"{rre._model_slug(model)}_seed{seed}.log")
    cmd = [sys.executable, "-m", "src.experiments.run_all", "--only", model, "--seed", str(seed)]
    if config:
        cmd += ["--config", config]
    print(f"[worker] start  {model!r} seed {seed}  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    ok = rc == 0 and seed in rre._csv_seeds(model)
    _mark_state(model, seed, ok, lock)
    print(f"[worker] {'OK   ' if ok else 'FAIL '} {model!r} seed {seed} exit {rc} in {(time.time() - t0) / 60:.1f} min",
          flush=True)
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
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATUS_FILE)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-runs", type=int, default=10, help="seeds per condition (default 10, as pre-registered)")
    ap.add_argument("--base-seed", type=int, default=43)
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)),
                    help="entanglement-sweep levels to run, e.g. 0,2,4,6 or 'all' (default 0,2,4,6)")
    ap.add_argument("--config", default="config/default_config.yaml")
    ap.add_argument("--workers", type=int, default=1,
                    help="runs executed at the same time (default 1); threads per run = logical CPUs // workers. "
                         "4 cores / 8 logical: try 2-4")
    ap.add_argument("--max-stalled-passes", type=int, default=5,
                    help="give up after this many passes in a row with no new completed run (0 = never)")
    ap.add_argument("--retry-wait", type=float, default=300.0, help="seconds between passes (default 300)")
    ap.add_argument("--allow-head-only", action="store_true", help="skip the forecast_path=generator check")
    ap.add_argument("--no-hypothesis-tests", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    models = build_conditions(parse_levels(args.levels))
    seeds = [args.base_seed + i for i in range(args.n_runs)]
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

    if not args.allow_head_only:
        bad = check_forecast_path(args.config)
        if bad:
            raise SystemExit("[study] refusing to start: " + "; ".join(bad) +
                             ". H4 is untestable unless the generator is on the forecast path in BOTH GAN sections "
                             "(set forecast_path: generator, or pass --allow-head-only).")
    apply_sweep_config(args.config)
    known = rre._known_models(args.config)       # a misspelled/missing condition would otherwise fail every pass
    if known is not None and [m for m in models if m not in known]:
        raise SystemExit(f"[study] unknown model name(s) {[m for m in models if m not in known]}; run_all knows {known}")
    keep_awake()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    pass_fn = None
    if args.workers > 1:
        w = args.workers
        pass_fn = lambda ms, n, b, c: _run_pass_parallel(ms, n, b, c, w)   # noqa: E731
    rc = run_study(models, args.n_runs, args.base_seed, args.config, args.max_stalled_passes, args.retry_wait,
                   run_pass=pass_fn)
    if rc != 0:
        sys.exit(rc)
    print("\n[study] ALL RUNS COMPLETE.", flush=True)
    if not args.no_hypothesis_tests:
        sys.exit(subprocess.run([sys.executable, "-m", "scripts.run_hypothesis_tests"]).returncode)


if __name__ == "__main__":
    main()
