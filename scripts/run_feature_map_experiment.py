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


def main() -> int:
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
    a = ap.parse_args()

    ld = lambda n: np.load(os.path.join(a.processed_dir, n))  # noqa: E731
    Xtr, ytr = _tail(ld("X_train.npy"), a.max_train_rows), _tail(ld("y_train.npy"), a.max_train_rows)
    Xva, yva = _tail(ld("X_val.npy"), a.max_eval_rows), _tail(ld("y_val.npy"), a.max_eval_rows)
    Xte, yte = _tail(ld("X_test.npy"), a.max_eval_rows), _tail(ld("y_test.npy"), a.max_eval_rows)

    rows, dm = [], []
    for seed in a.seeds:
        errs = {}
        for arm in ARMS:
            m = FeatureMapForecaster(arm, a.n_qubits, a.n_layers, a.entanglement, a.input_scale, seed)
            r = m.fit(Xtr, ytr, Xva, yva).evaluate(Xte, yte)
            errs[arm] = r.pop("errors")
            r["seed"] = seed
            rows.append(r)
            print(f"seed {seed} {arm:13s} skill={r['skill']:+.5f} rmse={r['rmse']:.6g} "
                  f"persistence={r['persistence_rmse']:.6g}", flush=True)
        for other in ("linear", "classical_rf"):
            d = diebold_mariano(errs[other], errs["quantum"])
            d.update({"seed": seed, "vs": other})
            dm.append(d)
    summary = {}
    for arm in ARMS:
        s = np.array([r["skill"] for r in rows if r["arm"] == arm])
        summary[arm] = {"skill_mean": float(s.mean()), "skill_sd": float(s.std(ddof=1)) if len(s) > 1 else None}
    out = {"args": vars(a), "per_run": rows, "dm_quantum_vs": dm, "summary": summary,
           "reading": "skill>0 beats persistence. Quantum 'advantage' = DM statistic > 0 (control has larger loss) "
                      "with p<0.05 consistently across seeds AND over classical_rf, not only over linear."}
    from src.utils.run_identity import save_json_timestamped
    save_json_timestamped(a.out, out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
