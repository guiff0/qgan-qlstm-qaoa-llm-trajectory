"""Temporal stability and the pre-registered row-level rule, from saved per-row predictions.

    python -m scripts.run_temporal_validation --experiment-id EXP-20261107-WINBOX-3fa9

Reads results/predictions/<experiment>/<condition>_seed<k>.npz (written by run_all after each GAN run) and writes
results/temporal_validation__<experiment>.json: per condition, persistence skill overall / by test block / by VIX
regime with a block-bootstrap CI, and the paired Diebold-Mariano rule (quantum vs the large and the matched classical).
Runs that have no predictions file are listed, not guessed. Within-test-period stability only: not rolling-origin.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import temporal_validation as tv  # noqa: E402

QUANTUM, LARGE, MATCHED = "qgan_llm", "classical_gan_llm", "classical_gan_llm_matched"


def load_predictions(directory: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*_seed*.npz"))):
        m = re.match(r"(.+)_seed(\d+)\.npz$", os.path.basename(path))
        if not m:
            continue
        z = np.load(path, allow_pickle=False)
        arrs = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        arrs["n_rows_test_shifted"] = meta.get("n_rows_test_shifted")
        arrs["_meta"] = meta
        out[(m.group(1), int(m.group(2)))] = arrs
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def analyse(preds: dict, n_blocks: int = 10, n_boot: int = 2000) -> dict:
    conds = sorted({c for c, _ in preds})
    res = {"design_note": ("within one chronological test period: stability across contiguous blocks and VIX regimes; "
                           "NOT rolling-origin validation (retraining on several origins is not part of this study)"),
           "conditions": {}, "mismatched_logged_rmse": sorted(f"{c} seed {s}" for (c, s), a in preds.items()
                                                              if a["_meta"].get("matches_logged_rmse") is False)}
    skill = {}
    for c in conds:
        seeds = sorted(s for cc, s in preds if cc == c)
        per_seed = {}
        for s in seeds:
            a = preds[(c, s)]
            sk = tv.overall_skill(a)
            skill[(c, s)] = sk["ratio"]
            per_seed[s] = {**sk, "blocks": tv.block_skill(a, n_blocks), "regimes": tv.regime_skill(a),
                           "bootstrap": tv.block_bootstrap_ratio(a, n_boot=n_boot, seed=s)}
        ratios = np.array([v["ratio"] for v in per_seed.values()])
        blk = np.array([b["ratio"] for v in per_seed.values() for b in v["blocks"]], dtype=float)
        reg = {}
        for r in ("low", "normal", "high"):
            vals = [x["ratio"] for v in per_seed.values() for x in v["regimes"] if x["regime"] == r and np.isfinite(x["ratio"])]
            reg[r] = {"mean_ratio": float(np.mean(vals)) if vals else None, "n_runs": len(vals)}
        res["conditions"][c] = {
            "n_seeds": len(seeds), "mean_ratio": float(ratios.mean()), "sd_ratio": float(ratios.std(ddof=1)) if len(ratios) > 1 else None,
            "seeds_beating_persistence": int((ratios < 1).sum()),
            "share_of_blocks_beating_persistence": float(np.mean(blk[np.isfinite(blk)] < 1)) if np.isfinite(blk).any() else None,
            "by_regime": reg, "per_seed": per_seed}
    res["advantage_rule"] = {}
    for ctrl in (LARGE, MATCHED):
        if QUANTUM not in conds or ctrl not in conds:
            continue
        shared = sorted({s for c, s in preds if c == QUANTUM} & {s for c, s in preds if c == ctrl})
        dm = {s: tv.dm_same_rows(preds[(QUANTUM, s)], preds[(ctrl, s)]) for s in shared}
        res["advantage_rule"][ctrl] = {
            "dm_by_seed": {s: {k: v for k, v in r.items() if k in ("status", "statistic", "p_value", "mean_loss_diff", "a_better", "message")}
                           for s, r in dm.items()},
            **tv.advantage_rule(dm, {s: skill[(QUANTUM, s)] for s in shared})}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--pred-dir", default=os.path.join("results", "predictions"))
    ap.add_argument("--n-blocks", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = os.path.join(a.pred_dir, a.experiment_id)
    preds = load_predictions(d)
    if not preds:
        print(f"no prediction files in {d}. They are written by runs started with the Nov-07 code "
              f"(src/utils/prediction_dump.py); runs from before it need their evaluation repeated.")
        return 2
    res = _clean(analyse(preds, a.n_blocks, a.n_boot))
    res["experiment_id"] = a.experiment_id
    out = a.out or os.path.join("results", f"temporal_validation__{a.experiment_id}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"{'condition':<34}{'seeds':>6}{'RMSE/persist':>14}{'<1 in':>8}{'blocks<1':>10}  low / normal / high")
    for c, v in res["conditions"].items():
        rg = " / ".join(f"{(v['by_regime'][r]['mean_ratio'] or float('nan')):.3f}" for r in ("low", "normal", "high"))
        print(f"{c:<34}{v['n_seeds']:>6}{v['mean_ratio']:>14.3f}{v['seeds_beating_persistence']:>5}/{v['n_seeds']:<2}"
              f"{(v['share_of_blocks_beating_persistence'] or 0):>10.0%}  {rg}")
    for ctrl, r in res["advantage_rule"].items():
        print(f"pre-registered rule vs {ctrl}: {'MET' if r.get('rule_met') else 'not met'} -- {r.get('reading', r.get('status'))}")
    if res["mismatched_logged_rmse"]:
        print("WARNING: recomputed RMSE differs from the logged one for:", ", ".join(res["mismatched_logged_rmse"]))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
