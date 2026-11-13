r"""Run ONLY Arm 1 (the inputs-only baseline) of the three-arm protocol, whether the other two arms are finished or still running.

    python -m scripts.run_naive_baseline --status                  # what is done / pending for ALL three arms, runs nothing
    python -m scripts.run_naive_baseline --dry-run                 # the (window, seed) cells it would fit
    python -m scripts.run_naive_baseline                           # fit Arm 1 on every missing cell, then print the pairing
    python -m scripts.run_naive_baseline --wait-for-others         # keep re-checking until Arm 2 and Arm 3 are complete
    .\scripts\run_naive_baseline.ps1                               # same, with the venv's python (Windows)

WHAT "NAIVE BASELINE" MEANS HERE (methodology figure, Arm 1)
    forecast head  y_hat(t+1) = w0' x_t + b0   on the PCA features x_t only   (arm name "linear")
    reference      persistence "next close = current close", skill 0 by construction (reported beside it)
It is fitted by the SAME class (src/baselines/feature_map_forecaster.py), on the SAME chronological walk-forward windows
(src/evaluation/walk_forward.py, same embargo, no look-ahead), with the SAME seeds, the SAME ridge grid and validation split,
and scored by the SAME five metrics (persistence skill, RMSE, adversarial ASR, forecast fidelity, latency; src/evaluation/
arm_metrics.py) as Arms 2 and 3. Nothing about it depends on the other arms having finished.

WHY IT CAN RUN BESIDE A LIVE STUDY
    * It never opens results/all_results.csv, never touches logs/study or the experiment manifest, and never calls
      run_full_study: it writes only results/naive_baseline/<experiment or 'legacy'>/naive_baseline.json (atomic rewrite
      after every cell, so it is safe to kill and relaunch; finished cells are never repeated).
    * It lowers its own CPU priority and pins BLAS to --threads (default 1) so it does not slow the running arms.
    * The window plan comes from the walk-forward file of the other arms (--like, default results/walk_forward_q8.json) when it
      exists, so the cells match theirs exactly; without it, the run_walk_forward defaults are used and the plan is recorded.

THE RIDGE ARM IS DETERMINISTIC: seeds only change the random circuit/feature parameters, so the baseline's metrics are the same
for every seed within a window (ASR/latency subsample seeds aside). The replicates that carry variance are the WINDOWS; the
output says so, and the compliance check (src/evaluation/protocol_compliance.py) reads it that way.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _pin_threads(n: int) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)


def _lower_priority() -> None:
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        else:
            os.nice(10)
    except Exception:  # noqa: BLE001  (priority is a courtesy, never a failure)
        pass


def out_path(experiment: str | None, root_dir: str = "results") -> str:
    return os.path.join(root_dir, "naive_baseline", experiment or "legacy", "naive_baseline.json")


def load_json(path: str):
    try:
        with open(path, encoding="utf8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def atomic_write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=float)
    os.replace(tmp, path)


def cells_done(rows, arm: str) -> set:
    return {(int(r["window"]), int(r["seed"])) for r in rows if r.get("arm") == arm and "window" in r}


def status(plan_cells: set, own_rows, other_json) -> dict:
    """Per arm: cells done / pending against the plan. Pending = not yet in that arm's file (running OR not started)."""
    other = (other_json or {}).get("per_run", [])
    out = {}
    for arm, rows in (("linear", own_rows), ("classical_rf", other), ("quantum", other)):
        d = cells_done(rows, arm) & plan_cells
        out[arm] = {"done": len(d), "pending": len(plan_cells - d), "total": len(plan_cells)}
    out["others_complete"] = out["classical_rf"]["pending"] == 0 and out["quantum"]["pending"] == 0
    return out


def pairing(own_rows, other_json, plan_cells: set) -> dict:
    """Arm 1 beside Arms 2 and 3 on the cells where ALL THREE exist (identical windows and seeds). Descriptive only: the
    frozen decision rule stays in src/evaluation/three_arm.py (it needs the paired errors the walk-forward run holds)."""
    import numpy as np
    from src.evaluation.arm_metrics import METRIC_KEYS
    rows = [r for r in (other_json or {}).get("per_run", []) if r.get("arm") in ("classical_rf", "quantum")]
    rows += [r for r in own_rows if r.get("arm") == "linear"]
    common = plan_cells
    for arm in ("linear", "classical_rf", "quantum"):
        common = common & cells_done(rows, arm)
    own = [r for r in own_rows if r.get("arm") == "linear"]
    out = {"n_common_cells": len(common), "arms": {}, "arm1_all_own_cells": {}}
    for k in METRIC_KEYS:
        v = np.array([r[k] for r in own if k in r and r[k] is not None], dtype=float)
        v = v[np.isfinite(v)]
        out["arm1_all_own_cells"][k] = float(v.mean()) if len(v) else None
    for arm in ("linear", "classical_rf", "quantum"):
        sel = [r for r in rows if r.get("arm") == arm and (int(r["window"]), int(r["seed"])) in common]
        out["arms"][arm] = {}
        for k in METRIC_KEYS:
            v = np.array([r[k] for r in sel if k in r and r[k] is not None], dtype=float)
            v = v[np.isfinite(v)]
            out["arms"][arm][k] = {"mean": float(v.mean()), "n": int(len(v))} if len(v) else {"mean": None, "n": 0}
    return out


def build_plan(args, other_json):
    """(windows, seeds, X, y, source). Taken from the other arms' walk-forward file when present."""
    import numpy as np
    from src.evaluation import walk_forward as wf
    ld = lambda n: np.load(os.path.join(args.processed_dir, n))  # noqa: E731
    X, y = wf.concat_chronological([(ld(f"X_{s}.npy"), ld(f"y_{s}.npy")) for s in ("train", "val", "test")])
    if other_json and other_json.get("windows"):
        windows, source = other_json["windows"], f"windows copied from {args.like}"
        seeds = args.seeds or sorted({int(s) for s in (other_json.get("args") or {}).get("seeds", [])}) or list(range(10))
    else:
        windows = wf.make_windows(len(y), args.windows, args.test_len, min_train=args.min_train, val_frac=args.val_frac,
                                  gap=args.gap, mode=args.mode, max_train_rows=args.max_train_rows)
        source, seeds = "run_walk_forward defaults (no walk-forward file of the other arms found)", args.seeds or list(range(10))
    wf.assert_no_leak(windows)
    return windows, [int(s) for s in seeds], X, y, source


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-id", default=os.environ.get("EXPERIMENT_ID") or None,
                    help="label the output folder (default env EXPERIMENT_ID, else the ACTIVE experiment, else 'legacy')")
    ap.add_argument("--like", default=os.path.join("results", "walk_forward_q8.json"),
                    help="walk-forward file of Arms 2 and 3: its windows and seeds are reused cell for cell")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--test-len", type=int, default=20_000)
    ap.add_argument("--min-train", type=int, default=20_000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    ap.add_argument("--max-train-rows", type=int, default=100_000)
    ap.add_argument("--n-qubits", type=int, default=8, help="input width shared by all arms (columns [:n_qubits] feed the maps)")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--attack-rows", type=int, default=None, help="default: the walk-forward file's value, else 300")
    ap.add_argument("--attack-eps", type=float, default=None, help="default: the walk-forward file's value, else 0.1")
    ap.add_argument("--threads", type=int, default=1, help="BLAS threads (default 1: leave the cores to the running arms)")
    ap.add_argument("--wait-for-others", action="store_true", help="after fitting Arm 1, re-check until Arms 2 and 3 are complete")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    _pin_threads(a.threads)
    _lower_priority()
    if not a.experiment_id:
        try:
            from src.utils import experiment as exp
            a.experiment_id = exp.get_active()
        except Exception:  # noqa: BLE001
            a.experiment_id = None
    path = out_path(a.experiment_id)
    other = load_json(a.like)
    # Arm 1 must be scored by the SAME protocol as Arms 2 and 3: take it from their file when they exist
    oargs = (other or {}).get("args") or {}
    if a.attack_rows is None:
        a.attack_rows = int(oargs.get("attack_rows") or 300)
    if a.attack_eps is None:
        a.attack_eps = float(oargs.get("attack_eps") or 0.1)
    for k in ("n_qubits", "n_layers"):
        if oargs.get(k) is not None and getattr(a, k) != oargs[k]:
            print(f"[naive-baseline] {k}={getattr(a, k)} differs from the walk-forward file's {oargs[k]}; using {oargs[k]}")
            setattr(a, k, int(oargs[k]))
    try:
        windows, seeds, X, y, source = build_plan(a, other)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot plan the baseline run: {exc}")
        return 2
    plan = {(int(w["window"]), s) for w in windows for s in seeds}
    from src.baselines.feature_map_forecaster import DEFAULT_RIDGE_GRID
    saved = load_json(path) or {}
    rows = saved.get("per_run", [])
    protocol = {"ridge_grid": list(DEFAULT_RIDGE_GRID), "attack_rows": a.attack_rows, "attack_eps": a.attack_eps}
    why_stale = None
    if rows and saved.get("ridge_grid") != protocol["ridge_grid"]:
        why_stale = "it was fitted with a different (or unrecorded) ridge grid"
    elif rows and saved.get("windows") != windows:
        why_stale = "its windows differ from the current plan"
    elif rows and any((saved.get("args") or {}).get(k) != protocol[k] for k in ("attack_rows", "attack_eps")):
        why_stale = "it was scored with a different attack protocol"
    elif rows and any(r.get("metric_seed") != r.get("seed") for r in rows):
        why_stale = "its rows predate the shared metric seed"
    if why_stale:
        stale = f"{path}.stale_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        os.replace(path, stale)
        print(f"[naive-baseline] existing results are NOT reused: {why_stale}. Moved to {stale}; Arm 1 is refitted on every cell.")
        rows = []

    print(f"[naive-baseline] experiment={a.experiment_id or 'legacy'}  {len(windows)} windows x {len(seeds)} seeds = {len(plan)} cells")
    print(f"[naive-baseline] plan: {source}")
    st = status(plan, rows, other)
    for arm, label in (("linear", "Arm 1 naive baseline (inputs only)"), ("classical_rf", "Arm 2 classical random features"),
                       ("quantum", "Arm 3 quantum circuit features")):
        s = st[arm]
        print(f"  {label:<38} {s['done']:>3}/{s['total']} done, {s['pending']} pending")
    if a.status:
        return 0
    todo = [(w, s) for w in windows for s in seeds if (int(w["window"]), s) not in cells_done(rows, "linear")]
    if a.dry_run:
        for w, s in todo:
            print(f"  would fit Arm 1: window {w['window']} seed {s}")
        print(f"[dry-run] {len(todo)} cell(s); nothing executed.")
        return 0

    from src.baselines.feature_map_forecaster import FeatureMapForecaster
    from src.evaluation import walk_forward as wf

    def save():
        atomic_write(path, {"arm": "linear", "role": "naive baseline (inputs only)", "experiment_id": a.experiment_id,
                            "updated_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "plan_source": source,
                            "windows": windows, "seeds": seeds, "args": vars(a), "ridge_grid": protocol["ridge_grid"], "per_run": rows,
                            "deterministic_arm_note": "ridge on x is deterministic: metrics repeat across seeds within a window "
                                                      "(attack/latency subsample aside); windows carry the variance"})

    t0 = time.time()
    from datetime import datetime as _dt
    _ts = lambda: _dt.now().strftime("[%Y-%m-%d %H:%M:%S]")  # noqa: E731
    for k, (w, seed) in enumerate(todo, 1):
        print(f"{_ts()} START cell {k}/{len(todo)} window {w['window']} seed {seed}", flush=True)
        Xtr, ytr, Xva, yva, Xte, yte = wf.slice_window(X, y, w)
        m = FeatureMapForecaster("linear", a.n_qubits, a.n_layers, "ring", 1.0, seed)
        r = m.fit(Xtr, ytr, Xva, yva).evaluate(Xte, yte, attack_rows=a.attack_rows, attack_eps=a.attack_eps, metric_seed=seed)
        r.pop("errors", None)
        r.update({"seed": seed, "window": int(w["window"])})
        rows.append(r)
        save()
        print(f"{_ts()} [naive-baseline] cell {k}/{len(todo)} window {w['window']} seed {seed}: skill={r['skill']:+.5f} rmse={r['rmse']:.6g} "
              f"asr={r['asr']:.1f}% fidelity={r['fidelity']:.3f} latency={r['latency_ms']:.4f}ms  [{time.time() - t0:.0f}s]", flush=True)
    save()
    while True:
        other = load_json(a.like)
        st = status(plan, rows, other)
        print(f"[naive-baseline] Arm 2: {st['classical_rf']['done']}/{st['classical_rf']['total']}  "
              f"Arm 3: {st['quantum']['done']}/{st['quantum']['total']}")
        if st["others_complete"] or not a.wait_for_others:
            break
        time.sleep(a.poll_seconds)
    pr = pairing(rows, other, plan)
    saved = load_json(path) or {}
    saved["pairing"], saved["others_complete"] = pr, st["others_complete"]
    atomic_write(path, saved)
    print(f"[naive-baseline] cells with all three arms: {pr['n_common_cells']}/{len(plan)}"
          + ("" if st["others_complete"] else "  (Arms 2/3 still running: pairing is PARTIAL; re-run this command later)"))
    if pr["n_common_cells"] == 0:
        print("  Arm 1 alone (all its cells): " + "  ".join(f"{k}={v:.5g}" for k, v in pr["arm1_all_own_cells"].items() if v is not None))
    for arm, mm in (pr["arms"].items() if pr["n_common_cells"] else []):
        print("  " + f"{arm:<13}" + "  ".join(f"{k}={v['mean']:.5g}" if v["mean"] is not None else f"{k}=n/a" for k, v in mm.items()))
    print(f"[naive-baseline] written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
