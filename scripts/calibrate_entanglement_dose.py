"""Stage 1 gate for the dose-response H2/M2 design. ANALYSIS ONLY: it reads runs, it does not train anything.

*** UNTESTED AGAINST A REAL RUN, AND NOT USABLE UNTIL entangler.py IS WIRED ***
src/quantum/entangler.py defines the "rzz:<theta>" topology but nothing calls it yet. The entangling block is copy-pasted in
circuits.py (x2), pqc_runner.py, gradient_obfuscation.py, encoding_fidelity.py, attacks/quantum_attacks.py and feature_map.py.
Wire all six, confirm rzz:0.0 emits no two-qubit gate at every site, and pass tests/test_pqc_runner.py on a machine with
PennyLane BEFORE producing the runs this script reads. Otherwise it would certify a circuit that is not the one that trains.

Workflow
  1. python -m scripts.calibrate_entanglement_dose --print-config     # ablation entries for the 5 doses (+ positive control)
  2. paste them into config/default_config.yaml `ablations:`; run the 6 conditions on PILOT seeds 100-104
       python -m scripts.run_full_study --new-experiment --base-seed 100 --n-runs 5 --conditions "<the six names>" --label "dose calibration pilot"
  3. python -m scripts.calibrate_entanglement_dose [--csv results/all_results.csv --experiment EXP-...]

Gates (both must pass before any confirmatory run; thresholds are arguments so they are frozen in the amendment, not here):
  G1 entropy dose, ALL THREE required:
       (a) Jonckheere-Terpstra, increasing, p < alpha on the pooled pilot runs;
       (b) the MEAN entropy curve over the five doses is strictly increasing;
       (c) rank correlation between dose index and S_E over all pilot runs: Spearman rho >= --min-rho and one-sided p < alpha.
     Per-seed strict ordering is REPORTED, not required: a single draw is noisy (in a random-weight simulation at the generator's
     init scale only ~2-30 % of single draws are strictly increasing even when the expected curve rises), and M2's a-path needs
     monotonicity in expectation. Also reports the within-dose SD of S_E (M2's b-path needs spread at fixed theta).
  G2 positive control: ASR(degraded generator) - ASR(theta = 0 generator) >= --pc-min-gap points in >= 4 of 5 seeds.
     ASR is read for these two conditions ONLY. The dose levels' ASR is never read, so the dose grid cannot be chosen on ASR.
The positive-control condition is the theta = 0 architecture with `untrained: true` (QGANLLM trains for zero epochs, so the generator
keeps its random initial weights; the forecast head is fitted on [X, G(X)] as usual). Whether that gap is >= the frozen threshold is
decided by the pilot, never assumed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.dose_response import jonckheere_terpstra  # noqa: E402

THETAS = [0.0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2]
N_QUBITS, N_LAYERS = 12, 4
_DOSE = re.compile(r"(?:^|_)qgan_llm_dose_t(\d)_rzz(?:_\d{8}T\d{6}Z)?$")
_PC = re.compile(r"(?:^|_)qgan_llm_dose_pc_untrained(?:_\d{8}T\d{6}Z)?$")


def dose_name(k: int) -> str:
    return f"QGAN-LLM (dose-T{k} rzz)"


PC_NAME = "QGAN-LLM (dose-PC untrained)"


def config_block() -> str:
    lines = ["  # >>> dose-calibration conditions (scripts/calibrate_entanglement_dose.py --print-config) >>>"]
    for k, th in enumerate(THETAS):
        lines += [f'  - name: "{dose_name(k)}"', f"    n_qubits: {N_QUBITS}", f"    n_layers: {N_LAYERS}",
                  f'    entanglement: "rzz:{th!r}"', f"    # theta = {th:.6f} rad (fixed, never trained)"]
    lines += [f'  - name: "{PC_NAME}"', f"    n_qubits: {N_QUBITS}", f"    n_layers: {N_LAYERS}",
              f'    entanglement: "rzz:{THETAS[0]!r}"',
              "    untrained: true          # zero GAN epochs: random initial generator, same head/attack/evaluation as the theta=0 arm",
              "  # <<< dose-calibration conditions <<<"]
    return "\n".join(lines)


def tag_runs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rid = out["run_id"].astype(str)
    out["dose"] = rid.map(lambda s: int(m.group(1)) if (m := _DOSE.search(s)) else np.nan)
    out["is_pc"] = rid.map(lambda s: bool(_PC.search(s)))
    return out


def gate_entropy(df: pd.DataFrame, min_rho: float = 0.7, alpha: float = 0.05) -> dict:
    d = df[df["dose"].notna()].dropna(subset=["entanglement_entropy", "seed"])
    doses = sorted(d["dose"].unique())
    if len(doses) < 5:
        return {"passed": False, "reason": f"need all 5 doses, found {len(doses)}"}
    groups = [d[d["dose"] == k]["entanglement_entropy"].to_numpy() for k in doses]
    jt = jonckheere_terpstra(groups, "increasing")
    from scipy import stats as _st
    rho_all = _st.spearmanr(d["dose"], d["entanglement_entropy"])
    rho_p_one = float(rho_all.pvalue / 2) if rho_all.statistic > 0 else float(1 - rho_all.pvalue / 2)
    mean_curve = [float(np.mean(g)) for g in groups]
    mean_increasing = bool((np.diff(mean_curve) > 0).all())
    per_seed = []
    for s, g in d.groupby("seed"):
        m = g.groupby("dose")["entanglement_entropy"].mean().reindex(doses)
        per_seed.append(bool(m.notna().all() and (np.diff(m.to_numpy()) > 0).all()))
    spread = {int(k): float(np.std(g, ddof=1)) if len(g) > 1 else float("nan") for k, g in zip(doses, groups)}
    a_ok = jt["status"] == "ok" and jt["p_one_sided"] < alpha
    c_ok = bool(rho_all.statistic >= min_rho and rho_p_one < alpha)
    ok = a_ok and mean_increasing and c_ok
    return {"passed": bool(ok), "criteria": {"jt_increasing": bool(a_ok), "mean_curve_strictly_increasing": mean_increasing,
                                             "spearman_rank_correlation": c_ok},
            "jt": jt, "spearman_rho": float(rho_all.statistic), "spearman_p_one_sided": rho_p_one, "min_rho": min_rho,
            "seeds_strictly_ordered_info_only": int(sum(per_seed)), "n_seeds": len(per_seed),
            "mean_entropy_by_dose": {int(k): m for k, m in zip(doses, mean_curve)},
            "within_dose_sd": spread, "b_path_estimable": bool(all(np.isfinite(v) and v > 0 for v in spread.values())),
            "b_path_note": "M2's b-path needs S_E to vary at fixed theta; a within-dose SD near 0 means it is not estimable"}


def gate_positive_control(df: pd.DataFrame, min_gap: float, min_seeds: int = 4) -> dict:
    a = df[df["dose"] == 0].dropna(subset=["asr", "seed"]).set_index("seed")["asr"]
    p = df[df["is_pc"]].dropna(subset=["asr", "seed"]).set_index("seed")["asr"]
    seeds = sorted(set(a.index) & set(p.index))
    if not seeds:
        return {"passed": False, "reason": "no seed has both the theta = 0 run and the positive-control run"}
    gaps = {int(s): float(p.loc[s] - a.loc[s]) for s in seeds}
    n_ok = sum(abs(g) >= min_gap for g in gaps.values())
    return {"passed": bool(n_ok >= min_seeds), "gap_by_seed": gaps, "seeds_meeting_gap": int(n_ok),
            "min_gap_points": min_gap, "min_seeds": min_seeds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "all_results.csv"))
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--pilot-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--pc-min-gap", type=float, default=10.0, help="ASR points; freeze in the amendment")
    ap.add_argument("--min-rho", type=float, default=0.7, help="G1 Spearman threshold; freeze in the amendment")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    if args.print_config:
        print(config_block())
        return 0
    df = pd.read_csv(args.csv)
    if args.experiment and "experiment_id" in df.columns:
        df = df[df["experiment_id"] == args.experiment]
    df = tag_runs(df)
    df = df[df["seed"].isin(args.pilot_seeds)]
    g1, g2 = gate_entropy(df, args.min_rho, args.alpha), gate_positive_control(df, args.pc_min_gap)
    res = {"generated_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "pilot_seeds": args.pilot_seeds,
           "thetas": THETAS, "G1_entropy_monotone": g1, "G2_positive_control": g2,
           "stage2_allowed": bool(g1["passed"] and g2["passed"])}
    path = os.path.join(ROOT, "results", "calibration_entropy_dose.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"G1 entropy monotone : {'PASS' if g1['passed'] else 'FAIL'}  {g1.get('reason', '') or g1.get('criteria', '')}")
    print(f"G2 positive control : {'PASS' if g2['passed'] else 'FAIL'}  {g2.get('reason', '')}")
    print(f"Stage 2 allowed     : {res['stage2_allowed']}\nwritten: {path}")
    return 0 if res["stage2_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
