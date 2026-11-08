"""
Repeated-runs experiment harness -- the piece every one of H1-H5's statistical
tests actually needs (a t-test / ANCOVA / correlation on a SINGLE observation
per condition has no within-group variance to test at all).

Runs the SAME model config N times with different seeds, appending each run's
row to results/all_results.csv exactly like a normal run_all.py invocation
(same run_id scheme, same columns), so scripts/run_hypothesis_tests.py consumes
the result unchanged. (config, seed) pairs are what the analysis treats as
independent observations (src/utils/run_identity.latest_per_config_seed).

What this version adds over the first one
  * validates --model against `run_all --list-models` BEFORE any training
    (a misspelled name used to "succeed": run_all ran zero models, exited 0,
    and the seed was recorded as completed);
  * confirms each run really appended a (model_type, seed) row to
    results/all_results.csv before marking the seed complete;
  * seeds already present in the CSV count as done even if the state file was
    lost (no accidental duplicate runs);
  * several models at once (--model "A,B,C"), run seed-major, so if you stop
    early every condition has the same number of runs (balanced groups);
  * --dry-run, --max-consecutive-failures, --status.

HONEST LIMITATION: this is infrastructure, not compute. A QGAN-LLM run is slow;
choose --n-runs for what your hardware can finish (8-10 per condition is the
target for H1/H3/H4; see the design notes in the chat / TODO.md).

Examples
    python -m scripts.run_repeated_experiment --model "Classical GAN-LLM" --n-runs 8 --base-seed 43
    python -m scripts.run_repeated_experiment \\
        --model "Classical GAN-LLM,QGAN-LLM,QGAN-LLM (no noise)" --n-runs 8 --base-seed 43
    python -m scripts.run_repeated_experiment --model "QGAN-LLM" --n-runs 8 --dry-run
    python -m scripts.run_repeated_experiment --model "QGAN-LLM" --n-runs 8 --status

Use the SAME --base-seed for every condition you will compare (paired seeds).
Re-run the identical command to resume/retry; nothing already done is repeated.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

RESULTS_CSV = os.path.join("results", "all_results.csv")

# Experiment scoping (src/utils/experiment.py). None = legacy behaviour: every row of the CSV counts and the
# state files are shared. Set by main() (--experiment-id / env EXPERIMENT_ID) or by scripts/run_full_study.py.
EXPERIMENT_ID: str | None = None


def set_experiment_id(eid: str | None) -> None:
    global EXPERIMENT_ID
    EXPERIMENT_ID = eid or None
    if eid:
        os.environ["EXPERIMENT_ID"] = eid        # run_all subprocesses inherit it and stamp their rows


# ---------------------------------------------------------------- state ----
def _state_path(tag: str) -> str:
    if EXPERIMENT_ID:                            # per-experiment state: a second study never inherits "done"
        return os.path.join("logs", "experiments", EXPERIMENT_ID, f"repeated_experiment_{tag}.json")
    return os.path.join("logs", f"repeated_experiment_{tag}.json")


def _load_state(tag: str) -> dict:
    path = _state_path(tag)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_seeds": [], "failed_seeds": []}


def _save_state(tag: str, state: dict) -> None:
    os.makedirs(os.path.dirname(_state_path(tag)), exist_ok=True)
    tmp = _state_path(tag) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, _state_path(tag))          # atomic: an interrupt can't corrupt the state file


def _default_tag(model: str) -> str:
    return model.replace(" ", "_").replace("(", "").replace(")", "")


# -------------------------------------------------------------- results ----
def _model_slug(name: str) -> str:
    # same slug rule as src/utils/run_identity.model_slug (kept inline: this script must run standalone)
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def _config_of_run_id(run_id: str) -> str:
    return re.sub(r"_\d{8}T\d{6}Z$", "", str(run_id))


def _csv_seeds(model: str, path: str = RESULTS_CSV, experiment_id: str | None = None) -> set[int]:
    """Seeds that already have a row in the results CSV for this model CONFIG (and, when an experiment id is
    set, for THAT experiment only; rows with no experiment_id belong to the experiment 'unassigned').

    Rows are matched on the run_id slug, NOT on model_type: every QGAN-LLM ablation (no noise,
    12-qubit, the entanglement-sweep levels, ...) is written with model_type == "QGAN-LLM", so
    model_type alone would count the base model's seeds as done for every ablation. model_type is
    only used for a CSV that has no run_id column.
    """
    seeds: set[int] = set()
    if not os.path.isfile(path):
        return seeds
    want = _model_slug(model)
    eid = experiment_id if experiment_id is not None else EXPERIMENT_ID
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("seed") in (None, ""):
                continue
            if eid is not None and ((row.get("experiment_id") or "").strip() or "unassigned") != eid:
                continue
            if row.get("run_id"):
                match = _config_of_run_id(row["run_id"]) == want
            else:
                match = row.get("model_type") == model
            if match:
                try:
                    seeds.add(int(float(row["seed"])))
                except ValueError:
                    pass
    return seeds


def _known_models(config: str | None) -> list[str] | None:
    """Model names run_all would accept, or None if they cannot be determined."""
    cmd = [sys.executable, "-m", "src.experiments.run_all", "--list-models"]
    if config:
        cmd += ["--config", config]
    try:
        # Popen (not subprocess.run) on purpose: the orchestration tests count/replace subprocess.run
        # calls, and this probe is not a training run.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, _ = proc.communicate(timeout=300)
        if proc.returncode != 0 or not out:
            return None
        for line in reversed(out.strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                return list(json.loads(line))
    except Exception:  # noqa: BLE001 - validation is best-effort, never fatal by itself
        return None
    return None


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help='Exact model name, or several separated by commas, e.g. "Classical GAN-LLM,QGAN-LLM"')
    ap.add_argument("--n-runs", type=int, required=True, help="Independent seeded runs PER model")
    ap.add_argument("--base-seed", type=int, default=0,
                    help="Seeds used are base_seed, base_seed+1, ..., base_seed+n_runs-1")
    ap.add_argument("--tag", default=None,
                    help="State-file tag for resumability (default: derived from --model; "
                         "with several models the model name is appended)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--experiment-id", default=None,
                    help="Group these runs under one experiment (EXP-YYYYMMDD-MACHINE-xxxx). Resume/done counts and "
                         "state then refer to this experiment only. Default: env EXPERIMENT_ID, else legacy.")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run and exit")
    ap.add_argument("--status", action="store_true", help="Print completed/remaining seeds and exit")
    ap.add_argument("--max-consecutive-failures", type=int, default=3,
                    help="Stop after this many failed runs in a row (default 3) instead of burning hours on a "
                         "systematic error. 0 disables.")
    args = ap.parse_args()

    set_experiment_id(args.experiment_id or os.environ.get("EXPERIMENT_ID") or None)
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if not models or args.n_runs < 1:
        ap.error("need at least one --model and --n-runs >= 1")

    known = _known_models(args.config)
    if known is not None:
        bad = [m for m in models if m not in known]
        if bad:
            print(f"[run_repeated_experiment] unknown model name(s): {bad}\n"
                  f"  available: {known}", file=sys.stderr)
            sys.exit(2)
    else:
        print("[run_repeated_experiment] WARNING: could not list models from run_all; "
              "skipping name validation.")

    seeds = [args.base_seed + i for i in range(args.n_runs)]
    tags = {m: (args.tag if (args.tag and len(models) == 1)
                else f"{args.tag}_{_default_tag(m)}" if args.tag else _default_tag(m)) for m in models}
    states = {m: _load_state(tags[m]) for m in models}

    def done(m: str) -> set[int]:
        return set(states[m]["completed_seeds"]) | _csv_seeds(m)

    plan = [(s, m) for s in seeds for m in models if s not in done(m)]   # seed-major => balanced groups

    print(f"[run_repeated_experiment] models={models} n_runs={args.n_runs} base_seed={args.base_seed} "
          f"experiment={EXPERIMENT_ID or 'legacy (no experiment id)'}")
    for m in models:
        d = done(m) & set(seeds)
        print(f"  {m!r}: {len(d)}/{len(seeds)} done, "
              f"{len([s for s in seeds if s not in d])} remaining (tag={tags[m]!r})")
    if args.status:
        return
    if args.dry_run:
        for s, m in plan:
            print(f"  would run: {m!r} seed={s}")
        print(f"[dry-run] {len(plan)} run(s) total; nothing executed.")
        return

    consecutive_failures = 0
    for k, (seed, model) in enumerate(plan, 1):
        print(f"\n=== {model!r} seed {seed}  (run {k}/{len(plan)}) ===")
        cmd = [sys.executable, "-m", "src.experiments.run_all", "--only", model, "--seed", str(seed)]
        if args.config:
            cmd += ["--config", args.config]
        had_csv = os.path.isfile(RESULTS_CSV)
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        ok = result.returncode == 0
        if ok and (had_csv or os.path.isfile(RESULTS_CSV)) and seed not in _csv_seeds(model):
            # exit code 0 but no (model, seed) row appeared: nothing was actually recorded
            print(f"[FAIL] seed {seed}: run_all exited 0 but no row for config {_model_slug(model)!r}, seed={seed} "
                  f"was added to {RESULTS_CSV}")
            ok = False

        st = states[model]
        if ok:
            if seed not in st["completed_seeds"]:
                st["completed_seeds"].append(seed)
            if seed in st["failed_seeds"]:
                st["failed_seeds"].remove(seed)
            consecutive_failures = 0
            print(f"[OK] {model!r} seed {seed} completed in {elapsed:.1f}s")
        else:
            if seed not in st["failed_seeds"]:
                st["failed_seeds"].append(seed)
            consecutive_failures += 1
            print(f"[FAIL] {model!r} seed {seed} (exit {result.returncode}) after {elapsed:.1f}s "
                  f"-- re-run this command to retry")
        _save_state(tags[model], st)

        if args.max_consecutive_failures and consecutive_failures >= args.max_consecutive_failures:
            print(f"\n[run_repeated_experiment] stopping: {consecutive_failures} consecutive failures "
                  f"(likely a systematic error, not bad luck). Fix it and re-run the same command.")
            sys.exit(1)

    print("\n[run_repeated_experiment] finished.")
    incomplete = False
    for m in models:
        d = done(m) & set(seeds)
        print(f"  {m!r}: {len(d)}/{len(seeds)} completed, {len(states[m]['failed_seeds'])} failed. "
              f"Rows in {RESULTS_CSV} with run_id slug {_model_slug(m)!r}, one per seed (column 'seed').")
        incomplete |= len(d) < len(seeds)
    if incomplete:
        print("Re-run this exact command to retry remaining/failed seeds.")
    print("Next: python -m scripts.run_hypothesis_tests")


if __name__ == "__main__":
    main()
