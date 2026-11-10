"""Dose-response rules for the calibration-gated H2/M2 design (numpy/scipy only).

* jonckheere_terpstra : ordered-alternative trend test across dose groups (normal approximation, tie-corrected).
* h2_dose_verdict     : SUPPORTED / CONTRADICTED / EQUIVALENT / INCONCLUSIVE / NOT_DECIDABLE, as frozen in the amendment.
* mediation_preconditions : M2 is undefined (NOT_DECIDABLE), not null, when there is no total effect or the a-path is not monotone.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def jonckheere_terpstra(groups, alternative: str = "decreasing") -> dict:
    """groups: list of 1-D arrays ordered from LOWEST to HIGHEST dose. alternative: 'increasing' | 'decreasing'."""
    g = [np.asarray(x, dtype=float) for x in groups]
    g = [x[np.isfinite(x)] for x in g]
    if len(g) < 3 or any(len(x) < 2 for x in g):
        return {"status": "insufficient", "reason": "need >= 3 dose groups with >= 2 runs each"}
    J = 0.0
    for i in range(len(g) - 1):
        for j in range(i + 1, len(g)):
            d = g[j][None, :] - g[i][:, None]          # later (higher-dose) minus earlier
            J += np.sum(d > 0) + 0.5 * np.sum(d == 0)
    n = np.array([len(x) for x in g], dtype=float)
    N = n.sum()
    allv = np.concatenate(g)
    _, t = np.unique(allv, return_counts=True)
    var = (N * (N - 1) * (2 * N + 5) - np.sum(n * (n - 1) * (2 * n + 5)) - np.sum(t * (t - 1) * (2 * t + 5))) / 72.0
    var += (np.sum(n * (n - 1) * (n - 2)) * np.sum(t * (t - 1) * (t - 2))) / (36.0 * N * (N - 1) * (N - 2))
    var += (np.sum(n * (n - 1)) * np.sum(t * (t - 1))) / (8.0 * N * (N - 1))
    mean = (N * N - np.sum(n * n)) / 4.0
    z = (J - mean) / np.sqrt(var)
    p = stats.norm.sf(z) if alternative == "increasing" else stats.norm.cdf(z)
    return {"status": "ok", "J": float(J), "z": float(z), "p_one_sided": float(p), "alternative": alternative,
            "direction_observed": "increasing" if z > 0 else "decreasing", "n_groups": len(g), "N": int(N)}


def tost_difference(a, b, margin: float, alpha: float = 0.05) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    tc = stats.t.ppf(1 - alpha, df)
    lo, hi = d - tc * se, d + tc * se
    return {"diff": float(d), "ci90": [float(lo), float(hi)], "margin": float(margin), "equivalent": bool(lo > -margin and hi < margin)}


def h2_dose_verdict(groups_by_dose, positive_control_gap: float | None, pc_min_gap: float, margin: float = 5.0,
                    alpha: float = 0.05) -> dict:
    """groups_by_dose: ASR arrays ordered low -> high entangling dose. positive_control_gap: ASR(degraded) - ASR(trained)."""
    if positive_control_gap is None or abs(positive_control_gap) < pc_min_gap:
        return {"verdict": "NOT_DECIDABLE", "reason": "positive control did not separate from the trained generator at this precision"}
    jt = jonckheere_terpstra(groups_by_dose, "decreasing")
    if jt["status"] != "ok":
        return {"verdict": "INCONCLUSIVE", "reason": jt["reason"]}
    if jt["p_one_sided"] < alpha:
        return {"verdict": "SUPPORTED", "jt": jt}
    jt_up = jonckheere_terpstra(groups_by_dose, "increasing")
    if jt_up["p_one_sided"] < alpha:
        return {"verdict": "CONTRADICTED", "jt": jt_up}
    eq = tost_difference(groups_by_dose[0], groups_by_dose[-1], margin, alpha)
    return {"verdict": "EQUIVALENT" if eq["equivalent"] else "INCONCLUSIVE", "jt": jt, "tost_ends": eq}


def mediation_preconditions(total_effect_p: float, a_path_monotone: bool, n_runs: int, min_runs: int = 30,
                            alpha: float = 0.05) -> dict:
    problems = []
    if not (total_effect_p < alpha):
        problems.append("no significant total effect: mediation is undefined, not null")
    if not a_path_monotone:
        problems.append("dose -> mediator path is not monotone")
    if n_runs < min_runs:
        problems.append(f"fewer than {min_runs} independent runs")
    return {"ok": not problems, "verdict_if_failed": "NOT_DECIDABLE", "problems": problems}
