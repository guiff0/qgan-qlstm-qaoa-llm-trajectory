"""Three-arm fair-comparison analysis: naive baseline, inputs-only, classical random features, quantum circuit features.

Arms (src/baselines/feature_map_forecaster.py):  linear = ridge on x | classical_rf = ridge on [x, phi_c(x)] | quantum = ridge on [x, phi_q(x)].
NAIVE BASELINE = persistence ("next close = current close"): skill 0 by construction. A quantum "advantage" over a control that
itself cannot beat persistence is not an advantage at forecasting, so this module reports every arm against persistence first.

Pure numpy/scipy. Frozen decision rules come from PREREGISTRATION.md (Q1): advantage needs the quantum arm to beat BOTH controls
(paired DM p < 0.05 in its favour in >= 80 % of seeds, >= 8 of 10) AND skill > 0; equivalence is a paired TOST at +/- 2 % of
persistence RMSE; anything else is inconclusive and never reported as a trend.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

ARMS = ("linear", "classical_rf", "quantum")
RMSE_MARGIN = 0.02            # frozen: 2 % of persistence RMSE
SEED_SHARE = 0.8              # frozen: 8 of 10
MIN_SEEDS = 10


def _ci(x, level=0.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    t = stats.t.ppf(0.5 + level / 2, len(x) - 1)
    return [float(m - t * se), float(m + t * se)]


def _by_seed(rows, arm, key):
    d = {r["seed"]: r[key] for r in rows if r["arm"] == arm}
    return d


def per_arm_vs_naive(rows) -> dict:
    out = {}
    for arm in ARMS:
        s = np.array(list(_by_seed(rows, arm, "skill").values()), float)
        rp = np.array([r["rmse"] / r["persistence_rmse"] for r in rows if r["arm"] == arm], float)
        if len(s) == 0:
            continue
        p = float(stats.ttest_1samp(s, 0.0).pvalue) if len(s) > 1 and s.std() > 0 else float("nan")
        out[arm] = {"n_seeds": int(len(s)), "skill_mean": float(s.mean()), "skill_ci95": _ci(s), "p_skill_ne_0": p,
                    "seeds_beating_persistence": int((s > 0).sum()), "rmse_over_persistence_mean": float(rp.mean()),
                    "beats_naive": bool(s.mean() > 0 and _ci(s)[0] > 0)}
    return out


def paired_vs_control(rows, dm, control: str, margin: float = RMSE_MARGIN, share: float = SEED_SHARE) -> dict:
    q, c = _by_seed(rows, "quantum", "rmse"), _by_seed(rows, control, "rmse")
    pers = _by_seed(rows, "quantum", "persistence_rmse")
    seeds = sorted(set(q) & set(c))
    if not seeds:
        return {"control": control, "status": "no shared seeds"}
    d = np.array([(q[s] - c[s]) / pers[s] for s in seeds], float)      # <0: quantum has lower RMSE, in persistence units
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if n > 1 and sd > 0 else float("nan")
    ci90 = _ci(d, 0.90)
    equiv = bool(n > 1 and ci90[0] > -margin and ci90[1] < margin)
    mine = [x for x in dm if x.get("vs") == control and "statistic" in x]
    fav = sum(1 for x in mine if x["p_value"] < 0.05 and x["statistic"] > 0)      # control has the larger loss
    unf = sum(1 for x in mine if x["p_value"] < 0.05 and x["statistic"] < 0)
    need = math.ceil(share * len(mine)) if mine else None
    return {"control": control, "n_seeds": n, "rmse_diff_over_persistence_mean": float(d.mean()),
            "ci90": ci90, "p_paired_t": p, "tost_equivalent_within_margin": equiv, "margin": margin,
            "dm_seeds_favouring_quantum": fav, "dm_seeds_favouring_control": unf, "dm_seeds_total": len(mine),
            "dm_seeds_needed": need,
            "advantage_rule_met": bool(need is not None and fav >= need),
            "control_better_rule_met": bool(need is not None and unf >= need)}


def verdict(rows, dm) -> dict:
    naive = per_arm_vs_naive(rows)
    vs = {c: paired_vs_control(rows, dm, c) for c in ("linear", "classical_rf")}
    n = min((v.get("n_seeds", 0) for v in vs.values()), default=0)
    notes = []
    if not any(v["beats_naive"] for v in naive.values()):
        notes.append("NO arm beats the naive persistence baseline: any difference between arms is a difference between "
                     "forecasters that are no better than 'no change'")
    q_skill_ok = naive.get("quantum", {}).get("skill_mean", -1) > 0
    if n < MIN_SEEDS:
        v = "UNDERPOWERED"; notes.append(f"{n} paired seeds < the {MIN_SEEDS} the pre-registration requires; descriptive only")
    elif all(x["advantage_rule_met"] for x in vs.values()) and q_skill_ok:
        v = "ADVANTAGE_SUPPORTED"
    elif any(x["control_better_rule_met"] for x in vs.values()):
        v = "CONTRADICTED"
    elif all(x["tost_equivalent_within_margin"] for x in vs.values()):
        v = "EQUIVALENT_TO_CONTROLS"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "versus_naive": naive, "versus_controls": vs, "notes": notes,
            "rule": "quantum beats BOTH controls (paired DM p<.05, >= 80% of seeds) AND skill>0; equivalence = paired TOST +/-2% of persistence RMSE"}


def audit(rows, args: dict, ridge_grid, tol: float = 0.05) -> dict:
    """Is the three-arm comparison balanced? Checks what the results file can show; the rest is stated, not assumed."""
    checks, na = [], []

    def chk(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    seeds = {a: sorted(_by_seed(rows, a, "skill")) for a in ARMS}
    chk("same seeds fed to all three arms", seeds["linear"] == seeds["classical_rf"] == seeds["quantum"], {a: len(s) for a, s in seeds.items()})
    ntest = {a: sorted({r["n_test"] for r in rows if r["arm"] == a}) for a in ARMS}
    chk("identical test rows (same n_test in every arm and seed)", ntest["linear"] == ntest["classical_rf"] == ntest["quantum"] and len(ntest["quantum"]) == 1, ntest)
    nf = {a: sorted({r["n_features"] for r in rows if r["arm"] == a}) for a in ARMS}
    chk("feature dimensions: classical_rf == quantum design width", nf["classical_rf"] == nf["quantum"], nf)
    nq, nl = int(args["n_qubits"]), int(args["n_layers"])
    from ..quantum.feature_map import ClassicalRandomFeatureMap
    circuit_params = nl * nq * 3
    classical_params = ClassicalRandomFeatureMap(nq, nq, circuit_params, seed=0).n_random_params
    chk(f"random-parameter budget within {tol:.0%} (circuit {circuit_params} vs classical {classical_params})",
        abs(classical_params - circuit_params) <= tol * circuit_params, {"circuit": circuit_params, "classical": classical_params})
    chk("tuning effort: one shared ridge grid, one validation split, argmin on validation only", len(ridge_grid) >= 1,
        {"grid": list(ridge_grid), "validation_fits_per_arm": len(ridge_grid)})
    edge = {a: sorted({r["ridge"] for r in rows if r["arm"] == a} & {min(ridge_grid), max(ridge_grid)}) for a in ARMS}
    chk("no arm sits on the edge of the tuning grid (an edge optimum means the arm is under-tuned)", not any(edge.values()), edge)
    chk("training budget: closed-form fit on the same chronological rows", True,
        {"max_train_rows": args.get("max_train_rows"), "max_eval_rows": args.get("max_eval_rows")})
    na.append({"dimension": "synthetic-sample count", "status": "N/A in this study",
               "reason": "no generator: 0 synthetic rows in every arm. The generator study (QGAN vs classical GAN) is audited by "
                         "src/evaluation/balance_audit.py (synthetic_ratio, eval_samples)"})
    na.append({"dimension": "loss weights", "status": "N/A in this study",
               "reason": "one unweighted squared loss on the residual target in every arm. Generator study: balance_audit checks alpha == mse_weight"})
    return {"checks": checks, "not_applicable": na, "all_passed": all(c["passed"] for c in checks)}
