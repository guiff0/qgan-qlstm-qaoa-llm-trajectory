"""Calibrate the H3 poisoning strength ONCE, before the rerun, on a positive control that involves no generator.

    python -m scripts.calibrate_poisoning                 # reads data/processed/X_test.npy, y_test.npy
    python -m scripts.calibrate_poisoning --seeds 100 101 102 103 104 --target 10

What it does: for each shift_sd in a grid it applies the poisoning attack to a ridge head trained on REAL rows only and
records how much that head's clean-data RMSE rises. It prints the smallest shift_sd whose median degradation reaches the
target (default 10 %, twice the H3 margin of 5 points). The pilot seeds must lie outside the confirmatory seed list
(43...) so no confirmatory run informs the strength. It does not look at noise_strength, the QGAN, or any result of H3.

Then: paste the printed shift_sd into config/default_config.yaml (poisoning.shift_sd), set calibrated: true, and record the
change with `python -m scripts.freeze_preregistration --record-deviation "Amendment 2: poisoning strength calibrated"`.
If no shift reaches the target the head has no structure to damage on these features: H3 remains not decidable and the
honest report says so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.poisoning_protocol import calibrate_shift  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config().get("poisoning", {}) or {}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg.get("pilot_seeds", [100, 101, 102, 103, 104]))
    ap.add_argument("--confirmatory-seeds", type=int, nargs="+", default=list(range(43, 63)),
                    help="seeds the study uses; the pilot seeds must not overlap them")
    ap.add_argument("--fraction", type=float, default=float(cfg.get("fraction", 0.2)))
    ap.add_argument("--target", type=float, default=float(cfg.get("pc_target_pct", 10.0)))
    ap.add_argument("--max-rows", type=int, default=int(cfg.get("pc_max_rows", 100000)))
    ap.add_argument("--escalate-fractions", type=float, nargs="*", default=None,
                    help="fractions tried in increasing order if the configured one cannot reach the target (default 0.3 0.4)")
    args = ap.parse_args()

    clash = sorted(set(args.seeds) & set(args.confirmatory_seeds))
    if clash:
        print(f"refusing: pilot seeds {clash} overlap the confirmatory seeds; calibration must not use them")
        return 2
    pdir = args.processed_dir or load_config()["data"]["processed_dir"]
    try:
        X, y = np.load(os.path.join(pdir, "X_test.npy")), np.load(os.path.join(pdir, "y_test.npy"))
    except FileNotFoundError as exc:
        print(f"cannot read the processed test arrays in {pdir}: {exc}\nrun scripts/prepare_data.py first")
        return 2
    # SAME recipe as the in-run H3 evaluation (poisoning_protocol.chronological_split): earlier half trains, later half is scored,
    # 1-row embargo, control head on the first pc_max_rows training rows. The old recipe (random sorted subset) measured a
    # different quantity, which is why 662 % here became 0.3 % in the runs (scripts/diagnose_poisoning_gap.py).
    from src.evaluation.poisoning_protocol import DEFAULT_FRACTIONS, chronological_split
    sp = chronological_split(len(X), args.max_rows)
    Xtr, ytr = np.asarray(X[sp["train"]][sp["pc"]], dtype=np.float32), np.asarray(y[sp["train"]][sp["pc"]], dtype=np.float64)
    Xev, yev = np.asarray(X[sp["eval"]], dtype=np.float32), np.asarray(y[sp["eval"]], dtype=np.float64)
    fractions = [args.fraction] + [f for f in (args.escalate_fractions or DEFAULT_FRACTIONS) if f > args.fraction]
    out = None
    for fr in fractions:        # raise `fraction` only when no shift in the grid reaches the target at the smaller one
        out = calibrate_shift(Xtr, ytr, Xev, yev, seeds=args.seeds, fraction=fr, target_pct=args.target,
                              ridge=float(cfg.get("ridge", 1.0)))
        out["chosen_fraction"] = fr if out["chosen_shift_sd"] is not None else None
        if out["chosen_shift_sd"] is not None:
            break
    out["recipe"] = "chronological_split (same function as the in-run protocol)"
    half = len(Xtr)
    out["generated_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out["n_rows_train"], out["n_rows_eval"] = int(len(Xtr)), int(len(Xev))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"poisoning_calibration_{out['generated_utc']}.json")
    with open(path, "w", encoding="utf8") as fh:
        json.dump(out, fh, indent=2)
    print(f"{'shift_sd':>9} {'median %':>10} {'min %':>9}")
    for r in out["curve"]:
        print(f"{r['shift_sd']:9.2f} {r['median_degradation_pct']:10.2f} {r['min_degradation_pct']:9.2f}")
    print(f"\n{out['note']}\nwritten: {path}")
    if out["chosen_shift_sd"] is not None:
        print(f"\nconfig/default_config.yaml ->  poisoning:\n    shift_sd: {out['chosen_shift_sd']}\n    fraction: {out['chosen_fraction']}\n    calibrated: true")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
