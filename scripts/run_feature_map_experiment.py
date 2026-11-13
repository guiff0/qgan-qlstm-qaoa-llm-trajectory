"""
Fair test of quantum advantage in forecasting: linear vs classical random
features vs PQC features, residual-over-persistence target, several seeds,
paired Diebold-Mariano tests on the same test rows. No torch/PennyLane.

Usage:
  python -m scripts.run_feature_map_experiment --seeds 0 1 2 3 4 \
      --n-qubits 8 --n-layers 3 --max-train-rows 300000 --max-eval-rows 200000
Writes results/feature_map_advantage.json. Simulation cost is ~1.2 ms/row at
8 qubits (single core), so subsample rows with --max-*-rows (chronological tail).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.baselines.feature_map_forecaster import ARMS, FeatureMapForecaster  # noqa: E402
from src.evaluation.advantage_diagnostics import diebold_mariano  # noqa: E402


def _tail(a, n):
    return a if n is None or len(a) <= n else a[-n:]


PLAN_KEYS = ("seeds", "n_qubits", "n_layers", "entanglement", "input_scale", "max_train_rows", "max_eval_rows", "processed_dir")


def is_current(path: str, a, ridge_grid: list) -> bool:
    """True only if `path` holds every (arm, seed) cell of THIS plan, fitted with the current ridge grid and the shared metric seed."""
    try:
        with open(path, encoding="utf8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return False
    if d.get("ridge_grid") != ridge_grid or any((d.get("args") or {}).get(k) != getattr(a, k, None) for k in PLAN_KEYS):
        return False
    have = {(r.get("arm"), r.get("seed")) for r in d.get("per_run", []) if r.get("metric_seed") == r.get("seed") and "ridge_status" in r}
    return {(arm, s) for arm in ARMS for s in a.seeds} <= have


def _ts() -> str:
    from datetime import datetime
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def main_argv(argv) -> int:
    return main(argv)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--entanglement", default="ring")
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--max-train-rows", type=int, default=300_000)
    ap.add_argument("--max-eval-rows", type=int, default=200_000)
    ap.add_argument("--out", default="results/feature_map_advantage.json")
    ap.add_argument("--dry-run", action="store_true", help="check the data and print the plan; fit nothing (fast preflight)")
    ap.add_argument("--skip-if-current", action="store_true",
                    help="do nothing when --out already holds a COMPLETE run of this exact plan with the current ridge grid "
                         "(used by run_full_study.ps1 so a relaunch does not refit everything)")
    a = ap.parse_args(argv)
    from src.baselines.feature_map_forecaster import DEFAULT_RIDGE_GRID
    if a.skip_if_current and is_current(a.out, a, list(DEFAULT_RIDGE_GRID)):
        print(f"[feature-map] {a.out} is complete for this plan and the current ridge grid: nothing to do (delete it to refit)")
        return 0

    ld = lambda n: np.load(os.path.join(a.processed_dir, n))  # noqa: E731
    Xtr, ytr = _tail(ld("X_train.npy"), a.max_train_rows), _tail(ld("y_train.npy"), a.max_train_rows)
    Xva, yva = _tail(ld("X_val.npy"), a.max_eval_rows), _tail(ld("y_val.npy"), a.max_eval_rows)
    Xte, yte = _tail(ld("X_test.npy"), a.max_eval_rows), _tail(ld("y_test.npy"), a.max_eval_rows)

    print(f"[feature-map] train {len(Xtr):,} rows, val {len(Xva):,}, test {len(Xte):,}; {len(a.seeds)} seeds x {len(ARMS)} arms; "
          f"{a.n_qubits} qubits, {a.n_layers} layers, {a.entanglement}")
    problems = []
    if min(len(Xtr), len(Xva), len(Xte)) < 100:
        problems.append("fewer than 100 rows in a split")
    if not (np.isfinite(Xtr).all() and np.isfinite(ytr).all()):
        problems.append("non-finite values in the training data")
    if Xtr.shape[1] < a.n_qubits:
        problems.append(f"only {Xtr.shape[1]} feature columns for {a.n_qubits} qubits")
    if len(set(a.seeds)) != len(a.seeds):
        problems.append("duplicate seeds")
    if problems:
        print("[feature-map] CANNOT RUN: " + "; ".join(problems))
        return 2
    if a.dry_run:
        print("[feature-map] dry run: data and plan are valid; nothing fitted")
        return 0
    rows, dm = [], []
    for seed in a.seeds:
        errs = {}
        print(f"{_ts()} START seed {seed}", flush=True)
        for arm in ARMS:
            m = FeatureMapForecaster(arm, a.n_qubits, a.n_layers, a.entanglement, a.input_scale, seed)
            r = m.fit(Xtr, ytr, Xva, yva).evaluate(Xte, yte, metric_seed=seed)
            errs[arm] = r.pop("errors")
            r["seed"] = seed
            rows.append(r)
            print(f"{_ts()} seed {seed} {arm:13s} skill={r['skill']:+.5f} rmse={r['rmse']:.6g} "
                  f"persistence={r['persistence_rmse']:.6g}", flush=True)
        for other in ("linear", "classical_rf"):
            d = diebold_mariano(errs[other], errs["quantum"])
            d.update({"seed": seed, "vs": other})
            dm.append(d)
    summary = {}
    for arm in ARMS:
        s = np.array([r["skill"] for r in rows if r["arm"] == arm])
        summary[arm] = {"skill_mean": float(s.mean()), "skill_sd": float(s.std(ddof=1)) if len(s) > 1 else None}
    out = {"args": vars(a), "ridge_grid": list(DEFAULT_RIDGE_GRID), "per_run": rows, "dm_quantum_vs": dm, "summary": summary,
           "reading": "skill>0 beats persistence. Quantum 'advantage' = DM statistic > 0 (control has larger loss) "
                      "with p<0.05 consistently across seeds AND over classical_rf, not only over linear."}
    from src.utils.run_identity import save_json_timestamped
    save_json_timestamped(a.out, out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
