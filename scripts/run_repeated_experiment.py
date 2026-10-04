"""
Repeated-runs experiment harness -- the piece every one of H1-H5's
statistical tests actually needs (a t-test/ANCOVA/correlation on a
SINGLE observation per condition has no within-group variance to test
at all). Runs the SAME model config N times with different seeds,
appending each run's row to results/all_results.csv exactly like a
normal run_all.py invocation would (same run_id scheme, same columns)
so scripts/run_hypothesis_tests.py can consume the result with no
changes on its end.

HONEST LIMITATION THIS SCRIPT DOES NOT SOLVE: it is infrastructure, not
compute. Training QGAN-LLM once already takes a severe, documented
amount of time at the full 3.7M-row/58k-batch/20-qubit configuration
(see TODO.md's quantum-training-scale item) -- running it 100-200
times (the n the manuscript's own inferential tests assume) is not
something this script can make fast. Use it with the training-scale
reduction features (once built -- see TODO.md) or hand-pick a smaller
--n-runs for a genuinely tractable repeated-runs experiment today,
rather than assuming this script alone makes the manuscript's n=100-200
claims achievable.

Resumable: re-running with the same --tag skips seeds already present
in the state file, so an interrupted repeated-runs experiment picks up
where it left off rather than restarting or silently duplicating runs.

    python -m scripts.run_repeated_experiment --model "Classical GAN-LLM" --n-runs 10
    python -m scripts.run_repeated_experiment --model "QGAN-LLM (no noise)" --n-runs 5 --base-seed 100
    python -m scripts.run_repeated_experiment --model "Classical LSTM" --n-runs 20 --tag lstm_ci_test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _state_path(tag: str) -> str:
    return os.path.join("logs", f"repeated_experiment_{tag}.json")


def _load_state(tag: str) -> dict:
    path = _state_path(tag)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_seeds": [], "failed_seeds": []}


def _save_state(tag: str, state: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(_state_path(tag), "w") as f:
        json.dump(state, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help='Exact model name, e.g. "Classical GAN-LLM"')
    ap.add_argument("--n-runs", type=int, required=True)
    ap.add_argument("--base-seed", type=int, default=0,
                     help="Seeds used are base_seed, base_seed+1, ..., base_seed+n_runs-1")
    ap.add_argument("--tag", default=None,
                     help="State-file tag for resumability (default: derived from --model)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    tag = args.tag or args.model.replace(" ", "_").replace("(", "").replace(")", "")
    state = _load_state(tag)
    seeds = [args.base_seed + i for i in range(args.n_runs)]
    remaining = [s for s in seeds if s not in state["completed_seeds"]]

    print(f"[run_repeated_experiment] model={args.model!r} n_runs={args.n_runs} "
          f"tag={tag!r} -- {len(state['completed_seeds'])} already done, "
          f"{len(remaining)} remaining")

    for seed in remaining:
        print(f"\n=== seed {seed} ({seeds.index(seed) + 1}/{len(seeds)}) ===")
        cmd = [sys.executable, "-m", "src.experiments.run_all", "--only", args.model, "--seed", str(seed)]
        if args.config:
            cmd += ["--config", args.config]
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        if result.returncode == 0:
            state["completed_seeds"].append(seed)
            print(f"[OK] seed {seed} completed in {elapsed:.1f}s")
        else:
            state["failed_seeds"].append(seed)
            print(f"[FAIL] seed {seed} exited with code {result.returncode} after {elapsed:.1f}s "
                  f"-- not retried automatically; re-run this script to retry")
        _save_state(tag, state)

    print(f"\n[run_repeated_experiment] done: {len(state['completed_seeds'])}/{args.n_runs} "
          f"completed, {len(state['failed_seeds'])} failed. "
          f"Rows are in results/all_results.csv under model_type={args.model!r}, "
          f"distinguished by the 'seed' column.")
    if len(state["completed_seeds"]) < args.n_runs:
        print("Re-run this exact command to retry remaining/failed seeds.")


if __name__ == "__main__":
    main()
