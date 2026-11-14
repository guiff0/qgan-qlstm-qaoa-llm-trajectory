"""Selector comparison: does it matter that QAOA picks the features? EXPLORATORY (designed after the seed-42 pilot).

    python -m scripts.qaoa_selector_comparison                    # 10 QAOA seeds + 2 deterministic controls, no LLM call
    python -m scripts.qaoa_selector_comparison --seeds 42 43 44

One pipeline, one step swapped:   16 candidate features -> [selector] -> selected features -> same ridge head -> forecast

    QAOA      the model's own selector (QUBO solved by 3-layer QAOA), one run per seed (the seed sets the circuit's start)
    classical the top-k candidates by |correlation with the target|          (deterministic)
    exact     the true optimum of the SAME QUBO, found by listing all 2^16 answers (deterministic)

Everything else is the QAOA-Enhanced pilot model unchanged: same rows, same target (next-bar change), same inputs (feature
changes), same head (closed-form ridge, penalty chosen on validation with "no head" as a candidate). Reported per arm: which
features, the QUBO value and its rank among all 65,536 answers, and RMSE / persistence on validation and on test.

This is NOT the three-arm methodology (run_walk_forward / run_naive_baseline / feature-map experiment): it imports nothing
from it, reads none of its files and writes only results/qaoa_selector_comparison.{json,md}. No hypothesis verdict reads it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.qaoa_solution_audit import all_qubo_values                  # noqa: E402
from src.baselines.qaoa_llm import QAOALLM                               # noqa: E402
from src.quantum.qaoa import QAOAFeatureSelector, QAOAFeatureSelectorConfig  # noqa: E402
from src.utils.config import load_config                                 # noqa: E402

BANNER = "EXPLORATORY: designed after the seed-42 pilot; not a pre-registered hypothesis and not part of any verdict."


def build_qubo(model_cfg: dict, X_train, y_train, seed: int = 42):
    """The QUBO the model builds on these rows (it does not depend on the seed): candidates, importance, Q, const, all values."""
    m = QAOALLM(model_cfg, seed=seed)
    X, target, _a, _b = m._frame(X_train, y_train)
    sd = float(np.std(target))
    c = m.config
    sel = QAOAFeatureSelector(QAOAFeatureSelectorConfig(
        n_candidates=c["n_candidates"], target_k=c["target_k"], n_layers=c["qaoa_layers"], budget_penalty=c["budget_penalty"],
        redundancy_penalty=c["redundancy_penalty"], redundancy_edge_threshold=c["redundancy_edge_threshold"],
        quantum_device=c["quantum_device"], n_steps=c["qaoa_steps"], learning_rate=c["qaoa_learning_rate"], seed=seed))
    cand, importance = sel._select_candidates(X, target / (sd if sd > 0 else 1.0))
    Q, const = sel._build_qubo(X[:, cand], importance)
    return {"cand": np.asarray(cand), "importance": np.asarray(importance), "Q": Q, "const": const,
            "values": all_qubo_values(Q, const), "k": int(c["target_k"]), "n_features": int(X.shape[1])}


def describe(q: dict, selected) -> dict:
    """QUBO value and rank of a feature set (features outside the 16 candidates cannot be valued: value None)."""
    cand = [int(v) for v in q["cand"]]
    sel = sorted(int(s) for s in selected)
    if any(s not in cand for s in sel):
        return {"selected": sel, "n_selected": len(sel), "qubo_value": None, "rank": None}
    n = len(cand)
    idx = sum(1 << (n - 1 - w) for w in range(n) if cand[w] in sel)
    v = float(q["values"][idx])
    return {"selected": sel, "n_selected": len(sel), "qubo_value": v,
            "rank": int(np.sum(q["values"] < v - 1e-12)) + 1}


def fit_and_score(model_cfg: dict, seed: int, splits: dict, fixed=None) -> dict:
    """Train the pilot model (its own QAOA when fixed is None, else the given feature set) and score validation and test."""
    m = QAOALLM(model_cfg, seed=seed)
    if fixed is not None:
        mask = np.zeros(splits["X_train"].shape[1], dtype=bool)
        mask[list(fixed)] = True
        m._select_features = lambda X, y, run_logger=None: SimpleNamespace(selected_mask=mask, qubo_objective=float("nan"))
    t0 = time.time()
    m.train(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])
    r = m.evaluate(splits["X_test"], splits["y_test"])
    hi = getattr(m, "head_info", {}) or {}
    return {"selected": sorted(int(i) for i in np.flatnonzero(m.selector_result.selected_mask)),
            "val_ratio": hi.get("val_rmse_over_no_change"), "test_ratio": float(r["rmse"] / r["persistence_rmse"]),
            "ridge_penalty": hi.get("penalty"), "head_is_no_change": bool(hi.get("no_change", False)),
            "seconds": round(time.time() - t0, 1)}


def summarise(rows: list[dict], classical: dict, exact: dict, n_bitstrings: int) -> dict:
    t = np.array([r["test_ratio"] for r in rows], dtype=float)
    ranks = [r["rank"] for r in rows if r["rank"] is not None]
    out = {"n_seeds": len(rows), "qaoa_test_ratio_mean": float(t.mean()), "qaoa_test_ratio_min": float(t.min()),
           "qaoa_test_ratio_max": float(t.max()), "qaoa_test_ratio_sd": float(t.std(ddof=1)) if len(t) > 1 else None,
           "qaoa_seeds_beating_no_change_on_test": int(np.sum(t < 1.0)),
           "qaoa_seeds_better_than_classical_on_test": int(np.sum(t < classical["test_ratio"])),
           "qaoa_seeds_better_than_exact_on_test": int(np.sum(t < exact["test_ratio"])),
           "qaoa_seeds_finding_the_exact_optimum": int(sum(1 for r in rows if r["selected"] == exact["selected"])),
           "qaoa_distinct_selections": len({tuple(r["selected"]) for r in rows}),
           "qaoa_rank_median": float(np.median(ranks)) if ranks else None, "qaoa_rank_best": min(ranks) if ranks else None,
           "qaoa_rank_worst": max(ranks) if ranks else None, "n_bitstrings": n_bitstrings,
           "qaoa_seeds_with_k_features": int(sum(1 for r in rows if r["n_selected"] == len(exact["selected"])))}
    for name, ref in (("classical", classical), ("exact", exact)):
        d = t - ref["test_ratio"]
        out[f"qaoa_minus_{name}_mean"] = float(d.mean())
        if len(d) > 1 and float(d.std(ddof=1)) > 0:
            from scipy import stats
            half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
            out[f"qaoa_minus_{name}_ci95"] = [float(d.mean() - half), float(d.mean() + half)]
        else:
            out[f"qaoa_minus_{name}_ci95"] = None
    return out


def to_markdown(res: dict) -> str:
    f = lambda v, nd=6: "n/a" if v is None else f"{v:.{nd}f}"          # noqa: E731
    ci = lambda c: "n/a" if not c else f"[{c[0]:+.2e}, {c[1]:+.2e}]"   # noqa: E731
    L = [f"**{BANNER}**", "", "Same 16 candidates, same head, same rows; only the selector differs. RMSE / persistence: below 1 beats no-change.",
         "", "| Selector | Seed | Features | n | QUBO value | Rank of 65,536 | Validation | Test |", "|---|---|---|---|---|---|---|---|"]
    for name, r in (("Classical (top-k by correlation)", res["classical"]), ("Exact QUBO optimum", res["exact"])):
        L.append(f"| {name} | - | {r['selected']} | {r['n_selected']} | {f(r['qubo_value'], 3)} | {r['rank']:,} | {f(r['val_ratio'])} | {f(r['test_ratio'])} |")
    for r in res["qaoa"]:
        rank = "n/a" if r["rank"] is None else f"{r['rank']:,}"
        L.append(f"| QAOA | {r['seed']} | {r['selected']} | {r['n_selected']} | {f(r['qubo_value'], 3)} | {rank} | {f(r['val_ratio'])} | {f(r['test_ratio'])} |")
    s = res["summary"]
    L += ["", f"QAOA over {s['n_seeds']} seeds: test ratio mean {f(s['qaoa_test_ratio_mean'])} (min {f(s['qaoa_test_ratio_min'])}, max {f(s['qaoa_test_ratio_max'])}); "
              f"{s['qaoa_seeds_beating_no_change_on_test']} seed(s) beat no-change on test.",
          f"QAOA minus classical: {s['qaoa_minus_classical_mean']:+.2e} (95% CI {ci(s['qaoa_minus_classical_ci95'])}); "
          f"QAOA minus exact: {s['qaoa_minus_exact_mean']:+.2e} (95% CI {ci(s['qaoa_minus_exact_ci95'])}). The seeds share one "
          f"dataset and gave only {s['qaoa_distinct_selections']} distinct selections, so read these intervals as a description of "
          f"seed-to-seed spread, not as a test.",
          f"QAOA found the exact optimum in {s['qaoa_seeds_finding_the_exact_optimum']} of {s['n_seeds']} seeds; median rank "
          f"{s['qaoa_rank_median']:,.0f} (best {s['qaoa_rank_best']:,}, worst {s['qaoa_rank_worst']:,}); "
          f"{s['qaoa_seeds_with_k_features']} seed(s) returned exactly {len(res['exact']['selected'])} features; "
          f"{s['qaoa_distinct_selections']} distinct selections."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=os.path.join("results", "qaoa_selector_comparison"), help="path stem; .json and .md")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    pdir, mcfg = cfg["data"]["processed_dir"], cfg["qaoa_llm"]
    splits = {f"{k}_{s}": np.load(os.path.join(pdir, f"{k}_{s}.npy"), mmap_mode="r" if k == "X" else None)
              for k in ("X", "y") for s in ("train", "val", "test")}
    q = build_qubo(mcfg, splits["X_train"], splits["y_train"])
    n, k, cand = len(q["cand"]), q["k"], [int(v) for v in q["cand"]]
    best = int(np.argmin(q["values"]))
    exact_sel = [cand[w] for w in range(n) if (best >> (n - 1 - w)) & 1]
    res = {"banner": BANNER, "candidates": cand, "target_k": k, "seeds": a.seeds, "qaoa": []}

    def save():
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out + ".json", "w", encoding="utf8") as fh:
            json.dump(res, fh, indent=2)
    for name, sel in (("classical", cand[:k]), ("exact", exact_sel)):       # candidates are sorted by |corr|: the first k
        res[name] = {**describe(q, sel), **fit_and_score(mcfg, a.seeds[0], splits, fixed=sel)}
        print(f"[{name}] {res[name]['selected']} QUBO {res[name]['qubo_value']:.3f} rank {res[name]['rank']:,} "
              f"val {res[name]['val_ratio']:.5f} test {res[name]['test_ratio']:.5f}", flush=True)
        save()
    for sd in a.seeds:
        r = fit_and_score(mcfg, sd, splits)
        r = {"seed": sd, **describe(q, r["selected"]), **{kk: v for kk, v in r.items() if kk != "selected"}}
        res["qaoa"].append(r)
        print(f"[qaoa seed {sd}] {r['selected']} QUBO {r['qubo_value']} rank {r['rank']} val {r['val_ratio']:.5f} "
              f"test {r['test_ratio']:.5f} ({r['seconds']} s)", flush=True)
        save()
    res["summary"] = summarise(res["qaoa"], res["classical"], res["exact"], len(q["values"]))
    save()
    md = to_markdown(res)
    with open(a.out + ".md", "w", encoding="utf8") as fh:
        fh.write(md)
    print("\n" + md + f"written: {a.out}.json, {a.out}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
