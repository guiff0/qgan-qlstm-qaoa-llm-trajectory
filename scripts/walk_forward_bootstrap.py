"""Cluster (window-block) bootstrap of the three-arm walk-forward comparison.

Reads results/walk_forward_q8.json (written by scripts/run_walk_forward.py) and
resamples the logged (window, seed) cells. It refits nothing and changes no
existing file: it only summarises numbers that are already on disk.

Resampling unit: the walk-forward window (the cluster / time block). Seeds
inside a window share the same test rows, so they are not independent; each
bootstrap draw resamples windows with replacement and, inside each drawn
window, seeds with replacement (two-stage cluster bootstrap). The statistic is
the mean paired difference in skill (quantum minus control), where
skill = 1 - MSE / MSE_persistence, and the mean RMSE ratio to persistence.

    python -m scripts.walk_forward_bootstrap            # 8-qubit file
    python -m scripts.walk_forward_bootstrap --in results/walk_forward_q8.json --draws 10000 --seed 0
"""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
import numpy as np

ARMS = ("linear", "classical_rf", "quantum")


def cells(per_run):
    out = {}
    for r in per_run:
        out.setdefault((int(r["window"]), int(r["seed"])), {})[r["arm"]] = r
    return {k: v for k, v in out.items() if all(a in v for a in ARMS)}


def bootstrap(per_run, draws=10000, seed=0, level=0.95):
    c = cells(per_run)
    windows = sorted({w for w, _ in c})
    seeds = {w: sorted(s for ww, s in c if ww == w) for w in windows}
    def val(w, s, arm, key): return float(c[(w, s)][arm][key])
    def ratio(w, s, arm): return val(w, s, arm, "rmse") / val(w, s, arm, "persistence_rmse")
    stats = {
        "skill_diff_quantum_minus_linear": lambda w, s: val(w, s, "quantum", "skill") - val(w, s, "linear", "skill"),
        "skill_diff_quantum_minus_classical_rf": lambda w, s: val(w, s, "quantum", "skill") - val(w, s, "classical_rf", "skill"),
        "skill_linear": lambda w, s: val(w, s, "linear", "skill"),
        "skill_classical_rf": lambda w, s: val(w, s, "classical_rf", "skill"),
        "skill_quantum": lambda w, s: val(w, s, "quantum", "skill"),
        "rmse_ratio_linear": lambda w, s: ratio(w, s, "linear"),
        "rmse_ratio_classical_rf": lambda w, s: ratio(w, s, "classical_rf"),
        "rmse_ratio_quantum": lambda w, s: ratio(w, s, "quantum"),
    }
    rng = np.random.default_rng(seed)
    table = {k: {w: np.array([f(w, s) for s in seeds[w]]) for w in windows} for k, f in stats.items()}
    point = {k: float(np.mean([t[w].mean() for w in windows])) for k, t in table.items()}
    boot = {k: np.empty(draws) for k in stats}
    for b in range(draws):
        ws = rng.choice(windows, size=len(windows), replace=True)
        idx = {i: rng.integers(0, len(seeds[w]), size=len(seeds[w])) for i, w in enumerate(ws)}
        for k, t in table.items():
            boot[k][b] = np.mean([t[w][idx[i]].mean() for i, w in enumerate(ws)])
    a = (1 - level) / 2
    res = {}
    for k in stats:
        lo, hi = np.quantile(boot[k], [a, 1 - a])
        res[k] = {"mean": point[k], "ci_low": float(lo), "ci_high": float(hi),
                  "share_draws_above_0": float(np.mean(boot[k] > 0)),
                  "per_window_mean": {str(w): float(table[k][w].mean()) for w in windows}}
    return {"n_windows": len(windows), "n_cells": len(c), "seeds_per_window": {str(w): len(seeds[w]) for w in windows},
            "draws": draws, "rng_seed": seed, "level": level, "unit": "window (cluster), seeds resampled within window",
            "statistics": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/walk_forward_q8.json")
    ap.add_argument("--out", default="results/walk_forward_bootstrap.json")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    src = json.load(open(a.inp))
    out = bootstrap(src["per_run"], a.draws, a.seed)
    out["source"] = a.inp
    out["created_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    for k, v in out["statistics"].items():
        print(f"{k:42s} mean {v['mean']:+.3e}  95% CI [{v['ci_low']:+.3e}, {v['ci_high']:+.3e}]")
    print("written:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
