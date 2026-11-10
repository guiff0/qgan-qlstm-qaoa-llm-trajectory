"""Does the three-arm conclusion survive when the optimisation resources are varied or tilted AGAINST the quantum arm?

    python -m scripts.run_three_arm_matrix --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 8 --out results/three_arm_matrix.json

Every cell runs the three arms of src/baselines/feature_map_forecaster.py on the same chronological rows and the same seeds and
applies the frozen rule of src/evaluation/three_arm.py. Two kinds of cell:
  MATCHED  one resource varied identically for all arms: training budget (fraction of the training rows), feature dimension
           (n_qubits, so circuit width AND classical feature width), tuning effort (size of the shared ridge grid).
  BRACKET  the classical control is given MORE than the quantum arm: classical feature width x2 / x4 (with proportionally more
           random parameters), classical ridge grid 4x finer. If an "advantage" survives only when the controls are starved,
           it is not an advantage of the circuit.
Synthetic-sample count and loss weights do not exist in this experiment (no generator, one unweighted loss): they are audited for
the generator study in src/evaluation/balance_audit.py and reported as N/A here, not varied.
Pure numpy; cost is dominated by the circuit simulation (~1.2 ms/row at 8 qubits), so subsample with --max-*-rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.baselines.feature_map_forecaster import DEFAULT_RIDGE_GRID, FeatureMapForecaster  # noqa: E402
from src.evaluation import three_arm  # noqa: E402
from src.evaluation.advantage_diagnostics import diebold_mariano  # noqa: E402
from src.quantum.feature_map import ClassicalRandomFeatureMap  # noqa: E402


def _grid(k: int):
    """k log-spaced ridge values spanning the default grid (k = 6 reproduces it)."""
    return DEFAULT_RIDGE_GRID if k == 6 else tuple(np.logspace(np.log10(min(DEFAULT_RIDGE_GRID)), np.log10(max(DEFAULT_RIDGE_GRID)), k))


def run_cell(data, seeds, n_qubits=8, n_layers=3, entanglement="ring", train_frac=1.0, grid_all=6, classical_width_mult=1,
             classical_grid=None, input_scale=1.0) -> dict:
    Xtr, ytr, Xva, yva, Xte, yte = data
    k = max(int(len(Xtr) * train_frac), 50)
    Xtr, ytr = Xtr[-k:], ytr[-k:]
    rows, dm = [], []
    for seed in seeds:
        errs = {}
        for arm in three_arm.ARMS:
            g = _grid(classical_grid if (classical_grid and arm != "quantum") else grid_all)
            m = FeatureMapForecaster(arm, n_qubits, n_layers, entanglement, input_scale, seed, ridge_grid=g)
            if arm == "classical_rf" and classical_width_mult != 1:
                P = n_layers * n_qubits * 3 * classical_width_mult
                m.fmap = ClassicalRandomFeatureMap(n_qubits, n_qubits * classical_width_mult, P, seed=seed)
            r = m.fit(Xtr, ytr, Xva, yva).evaluate(Xte, yte)
            errs[arm] = r.pop("errors"); r["seed"] = seed
            rows.append(r)
        for other in ("linear", "classical_rf"):
            d = diebold_mariano(errs[other], errs["quantum"]); d.update({"seed": seed, "vs": other}); dm.append(d)
    v = three_arm.verdict(rows, dm)
    return {"verdict": v["verdict"], "skill": {a: s["skill_mean"] for a, s in v["versus_naive"].items()},
            "beats_naive": {a: s["beats_naive"] for a, s in v["versus_naive"].items()},
            "quantum_minus_control_rmse_over_persistence": {c: x.get("rmse_diff_over_persistence_mean") for c, x in v["versus_controls"].items()},
            "advantage_rule_met": {c: x.get("advantage_rule_met") for c, x in v["versus_controls"].items()}, "notes": v["notes"]}


def cells(n_qubits: int):
    out = [("baseline", {})]
    out += [(f"MATCHED train_frac={f}", {"train_frac": f}) for f in (0.25, 0.5)]
    out += [(f"MATCHED n_qubits={q}", {"n_qubits": q}) for q in (4, 6) if q != n_qubits]
    out += [(f"MATCHED ridge_grid={k}", {"grid_all": k}) for k in (1, 3, 12)]
    out += [(f"BRACKET classical width x{w}", {"classical_width_mult": w}) for w in (2, 4)]
    out += [("BRACKET classical ridge grid 24 vs quantum 6", {"classical_grid": 24})]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--max-train-rows", type=int, default=100_000)
    ap.add_argument("--max-eval-rows", type=int, default=50_000)
    ap.add_argument("--out", default="results/three_arm_matrix.json")
    a = ap.parse_args()
    ld = lambda n, m: np.load(os.path.join(a.processed_dir, n))[-m:]  # noqa: E731
    data = (ld("X_train.npy", a.max_train_rows), ld("y_train.npy", a.max_train_rows), ld("X_val.npy", a.max_eval_rows),
            ld("y_val.npy", a.max_eval_rows), ld("X_test.npy", a.max_eval_rows), ld("y_test.npy", a.max_eval_rows))
    res = {}
    for name, kw in cells(a.n_qubits):
        kw = {"n_qubits": a.n_qubits, "n_layers": a.n_layers, **kw}
        res[name] = run_cell(data, a.seeds, **kw)
        r = res[name]
        print(f"{name:<48} {r['verdict']:<24} skill q={r['skill']['quantum']:+.5f} rf={r['skill']['classical_rf']:+.5f} lin={r['skill']['linear']:+.5f}", flush=True)
    wins = [n for n, r in res.items() if r["verdict"] == "ADVANTAGE_SUPPORTED"]
    brk = [n for n in wins if n.startswith("BRACKET")]
    summary = {"cells": len(res), "cells_with_advantage": wins, "advantage_survives_bracket": bool(brk),
               "robust_reading": ("advantage appears only in cells where the classical controls are not given extra resources"
                                  if wins and not brk else
                                  "no cell shows an advantage" if not wins else "advantage also survives when controls get more resources"),
               "not_varied": ["synthetic-sample count (no generator here)", "loss weights (single unweighted loss)"]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf8") as fh:
        json.dump({"args": vars(a), "cells": res, "summary": summary}, fh, indent=2, default=float)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
