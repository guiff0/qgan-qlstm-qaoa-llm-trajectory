"""Walk-forward (rolling-origin) three-arm comparison: baseline (persistence + ridge on x) / classical random features / PQC
features, K chronological windows, the SAME seeds fed to every arm. No torch / PennyLane.

  python -m scripts.run_walk_forward --dry-run                                    # print the window plan, fit nothing
  python -m scripts.run_walk_forward --windows 5 --test-len 20000 --seeds 0 1 2 3 4 --n-qubits 8

Cost: the quantum arm simulates ~1.2 ms/row at 8 qubits, so one window x seed costs ~ (train + val + test rows) x 1.2 ms.
Cap the history with --max-train-rows (chronological tail, default 100000 = ~2-3 min per fit).
Writes results/walk_forward_q<n_qubits>.json (+ a timestamped copy); scripts.run_hypothesis_tests reads it into the THREE-ARM LEDGER.
It is a stability read-out; it does not replace the pre-registered decision rule.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import walk_forward as wf  # noqa: E402


PLAN_KEYS = ("windows", "test_len", "min_train", "val_frac", "gap", "mode", "rolling_len", "max_train_rows", "seeds", "n_qubits",
             "n_layers", "entanglement", "input_scale", "attack_rows", "attack_eps")


def is_current(path: str, a, windows: list, ridge_grid: list, metric_keys) -> bool:
    """True only if `path` holds every (arm, window, seed) cell of THIS plan, every row has all five metrics, and it was fitted with
    the current ridge grid (an older grid or an older evaluation protocol means the file must be refitted, not reused)."""
    try:
        with open(path, encoding="utf8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return False
    old, now = d.get("args") or {}, vars(a)
    if d.get("ridge_grid") != ridge_grid or d.get("windows") != windows:
        return False
    if any(old.get(k) != now.get(k) for k in PLAN_KEYS if k in now):
        return False
    rows = d.get("per_run") or []
    need = {(arm, w["window"], s) for arm in ("linear", "classical_rf", "quantum") for w in windows for s in a.seeds}
    have = {(r.get("arm"), r.get("window"), r.get("seed")) for r in rows
            if all(r.get(k) is not None for k in metric_keys) and r.get("metric_seed") == r.get("seed")}
    return need <= have


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--test-len", type=int, default=20_000)
    ap.add_argument("--min-train", type=int, default=20_000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    ap.add_argument("--rolling-len", type=int, default=None)
    ap.add_argument("--max-train-rows", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--entanglement", default="ring")
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--attack-rows", type=int, default=300)
    ap.add_argument("--attack-eps", type=float, default=0.1)
    ap.add_argument("--out", default=None, help="default results/walk_forward_q<n_qubits>.json")
    ap.add_argument("--skip-if-current", action="store_true",
                    help="do nothing when --out already holds a COMPLETE run of this exact plan with the current ridge grid "
                         "(used by run_full_study.ps1 so a relaunch does not refit everything)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ld = lambda n: np.load(os.path.join(a.processed_dir, n))  # noqa: E731
    try:
        X, y = wf.concat_chronological([(ld(f"X_{s}.npy"), ld(f"y_{s}.npy")) for s in ("train", "val", "test")])
    except FileNotFoundError as exc:
        print(f"cannot read the processed arrays in {a.processed_dir}: {exc}")
        return 2
    try:
        windows = wf.make_windows(len(y), a.windows, a.test_len, min_train=a.min_train, val_frac=a.val_frac, gap=a.gap,
                                  mode=a.mode, rolling_len=a.rolling_len, max_train_rows=a.max_train_rows)
    except ValueError as exc:
        print(f"cannot plan the windows: {exc}")
        return 2
    print(f"{len(y):,} chronological rows; {a.windows} {a.mode} windows of {a.test_len:,} test rows, embargo {a.gap}")
    print(f"{'win':>3} {'train rows':>22} {'val rows':>22} {'test rows':>22}")
    for w in windows:
        print(f"{w['window']:>3} {w['train_start']:>10,}-{w['train_end']:<10,} {w['val_start']:>10,}-{w['val_end']:<10,} "
              f"{w['test_start']:>10,}-{w['test_end']:<10,}")
    if a.dry_run:
        return 0
    from src.baselines.feature_map_forecaster import DEFAULT_RIDGE_GRID
    from src.evaluation.arm_metrics import METRIC_KEYS
    path = a.out or os.path.join("results", f"walk_forward_q{a.n_qubits}.json")
    if a.skip_if_current and is_current(path, a, windows, list(DEFAULT_RIDGE_GRID), METRIC_KEYS):
        print(f"[walk-forward] {path} is complete for this plan and the current ridge grid: nothing to do (delete it to refit)")
        return 0
    per_run, dm = wf.run_windows(X, y, windows, a.seeds, n_qubits=a.n_qubits, n_layers=a.n_layers,
                                 entanglement=a.entanglement, input_scale=a.input_scale,
                                 attack_rows=a.attack_rows, attack_eps=a.attack_eps)
    summary = wf.summarise(per_run, dm, windows)
    out = {"args": vars(a), "ridge_grid": list(DEFAULT_RIDGE_GRID), "windows": windows, "per_run": per_run, "dm_quantum_vs": dm, "summary": summary,
           "reading": "Each window re-fits every arm on earlier history only and scores the next window. skill > 0 beats "
                      "persistence (the naive baseline). A result that holds in one window but not the others is a regime effect."}
    from src.utils.run_identity import save_json_timestamped
    save_json_timestamped(path, out)
    print(json.dumps({k: v for k, v in summary.items() if k != "windows"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
