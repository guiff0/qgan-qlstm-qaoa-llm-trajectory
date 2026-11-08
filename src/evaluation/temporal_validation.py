"""Temporal stability of the forecasts, from the saved per-row predictions (reviewer claim 4).

WHAT THIS IS AND IS NOT. The study trains once on a chronological train/validation split and tests on one later period.
Retraining on rolling or expanding origins would multiply an already 1,200-hour study and is not done. Within the single
test period this module reports:
  * persistence skill (RMSE / persistence RMSE, < 1 beats 'no change') overall, by contiguous test BLOCK, and by VIX regime;
  * a block-bootstrap confidence interval for the overall ratio;
  * row-level paired Diebold-Mariano tests between two conditions on the SAME rows (same seed => same rows), and the
    pre-registered consistency rule across seeds.
It is evidence about stability within one out-of-sample period and across market regimes. It is not rolling-origin
validation, and the manuscript must say so.
"""
from __future__ import annotations

import math

import numpy as np

from src.evaluation.advantage_diagnostics import diebold_mariano
from src.evaluation.regime_eval import regime_labels


def _ratio(sq_model, sq_persist) -> float:
    d = float(np.mean(sq_persist))
    return math.sqrt(float(np.mean(sq_model)) / d) if d > 0 else float("nan")


def errors(arrs: dict) -> tuple:
    e_m = np.asarray(arrs["y_true"], float) - np.asarray(arrs["y_pred"], float)
    e_p = np.asarray(arrs["y_true"], float) - np.asarray(arrs["y_persist"], float)
    return e_m, e_p


def overall_skill(arrs: dict) -> dict:
    e_m, e_p = errors(arrs)
    return {"n": int(len(e_m)), "rmse": float(np.sqrt(np.mean(e_m ** 2))),
            "persistence_rmse": float(np.sqrt(np.mean(e_p ** 2))), "ratio": _ratio(e_m ** 2, e_p ** 2)}


def block_skill(arrs: dict, n_blocks: int = 10, min_rows: int = 30) -> list:
    """Persistence skill in `n_blocks` equal contiguous stretches of the test period (by row position)."""
    e_m, e_p = errors(arrs)
    pos = np.asarray(arrs["row_index"], float)
    total = float(arrs.get("n_rows_test_shifted") or (pos.max() + 1))
    blk = np.minimum((pos / total * n_blocks).astype(int), n_blocks - 1)
    out = []
    for b in range(n_blocks):
        m = blk == b
        out.append({"block": b + 1, "n": int(m.sum()),
                    "ratio": _ratio(e_m[m] ** 2, e_p[m] ** 2) if m.sum() >= min_rows else float("nan")})
    return out


def regime_skill(arrs: dict, min_rows: int = 30) -> list:
    if "vix" not in arrs:
        return []
    e_m, e_p = errors(arrs)
    lab = regime_labels(arrs["vix"])
    return [{"regime": r, "n": int((lab == r).sum()),
             "ratio": _ratio(e_m[lab == r] ** 2, e_p[lab == r] ** 2) if (lab == r).sum() >= min_rows else float("nan")}
            for r in ("low", "normal", "high")]


def block_bootstrap_ratio(arrs: dict, block_len: int = 20, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Moving-block bootstrap of the RMSE/persistence ratio over the rows in time order."""
    e_m, e_p = errors(arrs)
    order = np.argsort(np.asarray(arrs["row_index"]))
    sm, sp = (e_m ** 2)[order], (e_p ** 2)[order]
    n = len(sm)
    if n < 2 * block_len:
        return {"status": "insufficient_data", "n": int(n)}
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_len)
    starts_max = n - block_len
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, starts_max + 1, n_blocks)
        idx = (st[:, None] + np.arange(block_len)[None, :]).reshape(-1)[:n]
        ratios[i] = _ratio(sm[idx], sp[idx])
    lo, hi = np.quantile(ratios, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"status": "ok", "ratio": _ratio(sm, sp), "ci_low": float(lo), "ci_high": float(hi), "level": level,
            "block_len": block_len, "n_boot": n_boot, "beats_persistence": bool(hi < 1.0)}


def dm_same_rows(arrs_a: dict, arrs_b: dict) -> dict:
    """Row-level Diebold-Mariano between two runs scored on identical rows. Positive statistic = a has the larger loss."""
    if not np.array_equal(arrs_a["row_index"], arrs_b["row_index"]):
        return {"status": "rows_differ", "message": "the two runs were not scored on the same rows"}
    ea, _ = errors(arrs_a)
    eb, _ = errors(arrs_b)
    res = diebold_mariano(ea, eb)
    if "error" in res:
        return {"status": "insufficient_data", "message": res["error"]}
    res["status"] = "ok"
    res["a_better"] = bool(res["mean_loss_diff"] < 0)
    return res


def advantage_rule(per_seed: dict, skill_by_seed: dict, alpha: float = 0.05, fraction: float = 0.8) -> dict:
    """Pre-registered rule for one comparator (PREREGISTRATION.md): the quantum arm beats the control on RMSE in
    >= `fraction` of the seeds with a two-sided paired DM p < alpha in the quantum arm's favour, AND its persistence
    skill is better than 1. `per_seed`: {seed: dm_same_rows result with a = quantum}; `skill_by_seed`: {seed: ratio}."""
    ok = {s: r for s, r in per_seed.items() if r.get("status") == "ok"}
    n = len(ok)
    if n == 0:
        return {"status": "insufficient_data", "n_seeds": 0}
    need = math.ceil(fraction * n)
    wins = [s for s, r in ok.items() if r["a_better"] and r["p_value"] < alpha]
    same_sign = max(sum(r["a_better"] for r in ok.values()), sum(not r["a_better"] for r in ok.values()))
    ratios = [skill_by_seed[s] for s in ok if s in skill_by_seed and np.isfinite(skill_by_seed[s])]
    skill_ok = bool(ratios) and float(np.mean(ratios)) < 1.0
    z = [r["statistic"] for r in ok.values() if r.get("statistic") is not None]
    from scipy import stats
    stouffer_p = float(2 * stats.norm.sf(abs(sum(z) / math.sqrt(len(z))))) if z else None
    return {"status": "ok", "n_seeds": n, "needed": need, "seeds_where_quantum_wins_significantly": len(wins),
            "seeds_same_sign": same_sign, "stouffer_p_all_seeds": stouffer_p,
            "mean_persistence_ratio_quantum": float(np.mean(ratios)) if ratios else None,
            "persistence_skill_positive": skill_ok,
            "rule_met": bool(len(wins) >= need and skill_ok),
            "reading": (f"quantum beats the control with p < {alpha} in {len(wins)} of {n} seeds (needed {need}); "
                        f"mean RMSE/persistence = "
                        f"{(float(np.mean(ratios)) if ratios else float('nan')):.3f} (needs < 1)")}
