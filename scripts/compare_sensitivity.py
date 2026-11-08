"""Do the primary conclusions change under the preprocessing variants?

    python -m scripts.compare_sensitivity --parent EXP-20261107-WINBOX-3fa9

Finds every experiment whose manifest names --parent and whose label is 'sensitivity:<variant>', compares the 12-qubit QGAN
with the matched classical generator on FID, RMSE, RMSE/persistence and ASR (Welch, 95% CI), and reports for each variant
whether the verdict category (QGAN lower / higher / no significant difference at alpha 0.05) matches the 'baseline' variant.
Proxy for the 20-qubit comparison, 5 seeds per condition: a robustness check, not a re-test of H1/H4.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import experiment as exp  # noqa: E402
from src.utils.run_identity import latest_per_config_seed  # noqa: E402

A_SLUG, B_SLUG = "qgan_llm_12_qubit", "classical_gan_llm_matched"
OUTCOMES = ("fid", "rmse", "rmse_over_persistence", "asr")


def _slug(rid):
    return re.sub(r"_\d{8}T\d{6}Z$", "", str(rid))


def welch(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return {"status": "insufficient_data", "n_a": int(len(a)), "n_b": int(len(b))}
    t = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    dfree = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else len(a) + len(b) - 2
    tc = float(stats.t.ppf(1 - alpha / 2, dfree))
    diff = float(a.mean() - b.mean())
    cat = "no significant difference" if not (t.pvalue < alpha) else ("QGAN lower" if diff < 0 else "QGAN higher")
    return {"status": "ok", "n_a": int(len(a)), "n_b": int(len(b)), "diff": diff, "ci_low": diff - tc * se,
            "ci_high": diff + tc * se, "p_value": float(t.pvalue), "category": cat}


def compare_experiment(df: pd.DataFrame) -> dict:
    d = latest_per_config_seed(df)
    d["slug"] = d["run_id"].map(_slug)
    if "rmse" in d.columns and "persistence_rmse" in d.columns:
        d["rmse_over_persistence"] = d["rmse"] / d["persistence_rmse"]
    a, b = d[d["slug"] == A_SLUG], d[d["slug"] == B_SLUG]
    return {o: welch(a[o].dropna().to_numpy(float), b[o].dropna().to_numpy(float)) if o in d.columns
            else {"status": "missing_column"} for o in OUTCOMES}


def compare(parent: str, csv_path: str = os.path.join("results", "all_results.csv"), root: str = exp.EXPERIMENT_ROOT) -> dict:
    df = pd.read_csv(csv_path)
    found = {}
    for m in exp.list_experiments(root):
        lab = m.get("label") or ""
        if m.get("parent_experiment") == parent and lab.startswith("sensitivity:"):
            sub, _ = exp.select_experiments(df, [m["experiment_id"]])
            found[lab.split(":", 1)[1]] = {"experiment_id": m["experiment_id"], "results": compare_experiment(sub)}
    base = found.get("baseline", {}).get("results")
    rep = {"parent": parent, "variants": found, "baseline_present": base is not None, "changes": {}}
    if base:
        for v, item in found.items():
            if v == "baseline":
                continue
            rep["changes"][v] = {o: {"baseline": base[o].get("category"), "variant": item["results"][o].get("category"),
                                     "changed": base[o].get("category") != item["results"][o].get("category")}
                                 for o in OUTCOMES if base[o].get("status") == "ok" and item["results"][o].get("status") == "ok"}
        rep["conclusions_robust"] = all(not c["changed"] for ch in rep["changes"].values() for c in ch.values()) if rep["changes"] else None
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = compare(a.parent)
    if not rep["variants"]:
        print(f"no sensitivity experiments found with parent {a.parent} (see scripts.run_sensitivity_study)")
        return 2
    for v, item in rep["variants"].items():
        print(f"\n[{v}]  {item['experiment_id']}")
        for o, r in item["results"].items():
            if r.get("status") == "ok":
                print(f"   {o:<24} diff {r['diff']:+.4g}  95% CI [{r['ci_low']:+.4g}, {r['ci_high']:+.4g}]  p={r['p_value']:.3f}  -> {r['category']}")
    for v, ch in rep["changes"].items():
        flips = [o for o, c in ch.items() if c["changed"]]
        print(f"\n{v}: {'category CHANGED for ' + ', '.join(flips) if flips else 'same conclusions as baseline'}")
    if rep.get("conclusions_robust") is not None:
        print("\nPRIMARY CONCLUSIONS ROBUST TO PREPROCESSING:", "YES" if rep["conclusions_robust"] else "NO (see the variants above)")
    out = a.out or os.path.join("results", f"sensitivity__{a.parent}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
