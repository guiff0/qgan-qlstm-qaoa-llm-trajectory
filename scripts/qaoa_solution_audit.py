"""How good is the feature set QAOA returned? Exact check, DESCRIPTIVE.

    python -m scripts.qaoa_solution_audit --selected 1,2,3,11,16,17,21,26      # indices from the run log / results row

The QAOA-Enhanced LLM pilot solves one small optimisation problem (a QUBO over 16 pre-filtered candidate features). With
16 binary variables there are only 65,536 possible answers, so the true optimum can be found by listing all of them. This
script rebuilds the SAME problem the model built (same rows, same target frame, same penalties), lists every answer, and
reports where QAOA's answer ranks. It trains nothing and reads no test data; it is not part of any hypothesis verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.qaoa_llm import QAOALLM                               # noqa: E402
from src.quantum.qaoa import QAOAFeatureSelector, QAOAFeatureSelectorConfig  # noqa: E402
from src.utils.config import load_config                                 # noqa: E402


def all_qubo_values(Q: np.ndarray, const: float) -> np.ndarray:
    """QUBO value of every bitstring; index = the integer whose bit (n-1-w) is variable w (PennyLane's wire order)."""
    n = len(Q)
    idx = np.arange(2 ** n, dtype=np.int64)
    bits = ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.float64)     # (2^n, n)
    Qs = np.triu(Q, 1)
    return bits @ np.diag(Q) + np.einsum("bi,ij,bj->b", bits, Qs, bits) + const


def audit(X_train, y_train, model_cfg: dict, selected: list[int], seed: int = 42) -> dict:
    m = QAOALLM(model_cfg, seed=seed)
    if m.target_frame == "delta":
        X, target, _a, _b = m._frame(X_train, y_train)
        sd = float(np.std(target)); y = target / (sd if sd > 0 else 1.0)
    else:
        X, y = np.asarray(X_train), np.asarray(y_train)
    c = m.config
    sel = QAOAFeatureSelector(QAOAFeatureSelectorConfig(
        n_candidates=c["n_candidates"], target_k=c["target_k"], n_layers=c["qaoa_layers"], budget_penalty=c["budget_penalty"],
        redundancy_penalty=c["redundancy_penalty"], redundancy_edge_threshold=c["redundancy_edge_threshold"],
        quantum_device=c["quantum_device"], n_steps=c["qaoa_steps"], learning_rate=c["qaoa_learning_rate"], seed=seed))
    cand, importance = sel._select_candidates(X, y)
    Q, const = sel._build_qubo(X[:, cand], importance)
    vals = all_qubo_values(Q, const)
    n = len(cand)
    missing = [int(s) for s in selected if int(s) not in set(int(v) for v in cand)]
    x = np.array([int(int(ci) in set(selected)) for ci in cand])
    q_val = float(sel._qubo_value(Q, const, x.astype(bool)))
    best = int(np.argmin(vals))
    best_x = np.array([(best >> (n - 1 - w)) & 1 for w in range(n)])
    order = np.sort(vals)
    k = int(c["target_k"])
    greedy = np.zeros(n, dtype=int); greedy[:k] = 1                        # candidates are sorted by |corr|: the top k
    return {
        "frame": {"target": m.target_frame, "inputs": m.feature_frame}, "n_candidates": n, "n_bitstrings": int(2 ** n),
        "candidates": [int(v) for v in cand], "importance_abs_corr": [float(v) for v in importance],
        "qaoa": {"selected": sorted(int(s) for s in selected), "n_selected": int(x.sum()), "qubo_value": q_val,
                 "rank_among_all": int(np.searchsorted(order, q_val - 1e-12)) + 1,
                 "importance_captured": float(importance[x.astype(bool)].sum()), "selected_outside_candidates": missing},
        "exact_optimum": {"selected": sorted(int(cand[w]) for w in range(n) if best_x[w]), "n_selected": int(best_x.sum()),
                          "qubo_value": float(vals[best]), "importance_captured": float(importance[best_x.astype(bool)].sum())},
        "top_k_by_correlation": {"selected": sorted(int(v) for v in cand[:k]), "qubo_value": float(sel._qubo_value(Q, const, greedy.astype(bool))),
                                 "importance_captured": float(importance[:k].sum())},
        "worst_qubo_value": float(order[-1]), "median_qubo_value": float(np.median(vals)),
        "gap_to_optimum": q_val - float(vals[best]),
        "share_of_bitstrings_better_than_qaoa": float(np.mean(vals < q_val - 1e-12)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True, help="comma-separated feature indices QAOA selected (from the run log)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=os.path.join("results", "qaoa_solution_audit.json"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    pd_ = cfg["data"]["processed_dir"]
    X, y = np.load(os.path.join(pd_, "X_train.npy"), mmap_mode="r"), np.load(os.path.join(pd_, "y_train.npy"))
    out = audit(X, y, cfg["qaoa_llm"], [int(s) for s in a.selected.split(",") if s.strip()], a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf8") as fh:
        json.dump(out, fh, indent=2)
    q, e, t = out["qaoa"], out["exact_optimum"], out["top_k_by_correlation"]
    print(f"QUBO over {out['n_candidates']} candidates ({out['n_bitstrings']:,} possible answers); lower is better")
    print(f"  QAOA answer        : {q['selected']}  value {q['qubo_value']:.4f}  rank {q['rank_among_all']:,} of {out['n_bitstrings']:,} "
          f"({out['share_of_bitstrings_better_than_qaoa']:.2%} of all answers are better)")
    print(f"  exact optimum      : {e['selected']}  value {e['qubo_value']:.4f}")
    print(f"  top-{len(t['selected'])} by correlation: {t['selected']}  value {t['qubo_value']:.4f}")
    print(f"  gap QAOA - optimum : {out['gap_to_optimum']:.4f}   (median answer {out['median_qubo_value']:.2f}, worst {out['worst_qubo_value']:.2f})")
    print(f"  importance captured (sum |corr| with the target): QAOA {q['importance_captured']:.4f}, optimum {e['importance_captured']:.4f}, "
          f"top-k {t['importance_captured']:.4f}")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
