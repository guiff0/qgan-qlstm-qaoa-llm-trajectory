"""Paired-by-seed comparison of two conditions (reviewer claim 3).

Seed k fixes the generator's initial weights AND the evaluation subsample (src/utils/generator_path.forecast_eval_indices
uses the run seed), so QGAN seed 43 and classical seed 43 are scored on the same rows. Differences by seed therefore
remove the between-row noise that an unpaired Welch test leaves in. The unit is still the training run; only runs that
share a seed are paired, and only within ONE experiment (run_hypothesis_tests refuses to pool experiments for this).

Reported for outcome `col`, difference d_k = a_k - b_k:
  paired t test, 95% CI of the mean difference, Cohen's d_z, Wilcoxon signed-rank (n >= 6), exact sign test,
  and the pre-registered consistency rule: the sign of d_k is the same in at least `rule_fraction` of the seeds
  (8 of 10), with a paired TOST when an equivalence margin is given.
Row-level Diebold-Mariano tests need per-row errors: they are computed by scripts/run_temporal_validation.py from the
saved predictions (results/predictions/<experiment>/), not here.
"""
from __future__ import annotations

import math

import numpy as np


def _seed_means(rows, col):
    t = rows.dropna(subset=["seed", col]).copy()
    t["seed"] = t["seed"].astype(int)
    return t.groupby("seed")[col].mean()


def paired_by_seed(a_rows, b_rows, col: str, *, margin: float | None = None, alpha: float = 0.05,
                   rule_fraction: float = 0.8, min_pairs: int = 3) -> dict:
    from scipy import stats
    if "seed" not in a_rows.columns or "seed" not in b_rows.columns or col not in a_rows.columns:
        return {"status": "insufficient_data", "message": f"need 'seed' and '{col}' columns in both groups"}
    a, b = _seed_means(a_rows, col), _seed_means(b_rows, col)
    seeds = sorted(set(a.index) & set(b.index))
    n = len(seeds)
    out = {"status": "ok", "outcome": col, "n_pairs": n, "seeds": seeds,
           "unpaired_seeds": {"a_only": sorted(set(a.index) - set(b.index)), "b_only": sorted(set(b.index) - set(a.index))}}
    if n < min_pairs:
        out.update(status="insufficient_data", message=f"need >= {min_pairs} shared seeds, have {n}")
        return out
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    tcrit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    if se > 0:
        t_stat = mean / se
        p = float(2 * stats.t.sf(abs(t_stat), n - 1))
    else:
        t_stat, p = (0.0 if mean == 0 else math.copysign(math.inf, mean)), (1.0 if mean == 0 else 0.0)
    n_neg, n_pos, n_tie = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    k_same = max(n_neg, n_pos)
    out.update(mean_diff=mean, sd_diff=sd, t_statistic=float(t_stat), df=n - 1, p_value=p,
               ci_low=mean - tcrit * se, ci_high=mean + tcrit * se,
               cohens_dz=(mean / sd) if sd > 0 else None,
               direction="a < b" if mean < 0 else "a > b" if mean > 0 else "no difference",
               per_seed_diff={int(s): float(x) for s, x in zip(seeds, d)},
               sign_count={"a_lower": n_neg, "a_higher": n_pos, "ties": n_tie})
    nn = n_neg + n_pos
    out["sign_test_p"] = float(stats.binomtest(k_same, nn, 0.5).pvalue) if nn else 1.0
    need = math.ceil(rule_fraction * n)
    out["consistency_rule"] = {"same_sign_seeds": k_same, "needed": need, "fraction": rule_fraction,
                               "met": bool(k_same >= need and nn > 0),
                               "reading": (f"the sign of the seed-wise difference is the same in {k_same} of {n} seeds "
                                           f"(rule: at least {need})")}
    if n >= 6:
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            out["wilcoxon_p"] = None
    if margin is not None and margin > 0:
        if se > 0:
            p_lo = float(stats.t.sf((mean + margin) / se, n - 1))       # H0: diff <= -margin
            p_hi = float(stats.t.cdf((mean - margin) / se, n - 1))      # H0: diff >= +margin
            tc90 = float(stats.t.ppf(1 - alpha, n - 1))
            ci90 = (mean - tc90 * se, mean + tc90 * se)
        else:
            p_lo = p_hi = 0.0 if abs(mean) < margin else 1.0
            ci90 = (mean, mean)
        p_tost = max(p_lo, p_hi)
        out["equivalence"] = {"margin": margin, "p_tost": p_tost, "equivalent": bool(p_tost < alpha),
                              "ci_level": 1 - 2 * alpha, "ci_low": ci90[0], "ci_high": ci90[1],
                              "reading": ("EQUIVALENT: seed-wise difference lies within +/- margin" if p_tost < alpha
                                          else "equivalence NOT shown (not the same as 'different')")}
    return out
