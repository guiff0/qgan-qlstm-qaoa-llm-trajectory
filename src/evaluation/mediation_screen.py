"""EXPLORATORY mediator screen with a discovery / replication split. It never feeds a verdict.

WHY IT IS BUILT THIS WAY
  M2 asks one pre-specified question: does measured entanglement entropy mediate level -> ASR? On the 39-run sweep the
  answer is "no evidence" (indirect effect -0.069, 95 % CI [-0.207, 0.037]), and the TOTAL effect level -> ASR is itself
  null (Spearman rho = 0.02, p = 0.91). Trying other mediators on the same runs until one has a CI that excludes 0 is a
  garden of forking paths: with k candidates at 95 %, about 0.05 k "hits" are noise. Measured on the sweep: of 14
  candidates, `eer` excluded 0 in the first half of the seeds and did not in the second half or on all runs.

  So this screen (a) lists the candidates BEFORE it looks, (b) fits them on a discovery half of the seeds, (c) corrects
  the discovery step for the number of candidates (Holm over a bootstrap p), (d) requires the SAME sign and a CI that
  excludes 0 on the held-out replication half, and (e) labels every output EXPLORATORY. Anything that survives is a
  hypothesis for a NEW, pre-registered run with fresh seeds. It is not evidence for M2 on this data.
"""
from __future__ import annotations

import numpy as np

from .mediation import mediation_analysis

DEFAULT_CANDIDATES = ["entanglement_entropy", "purity", "qsfr", "eer", "nlcs", "ctr", "m1_efi", "rmse", "fid", "mmd",
                      "wasserstein", "mode_coverage", "latency_mean_ms", "poisoning_rmse_degradation_pct"]


def _z(a):
    a = np.asarray(a, dtype=float)
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else a * 0.0


def _indirect_p(iv, m, dv, n_bootstrap, seed):
    """Two-sided bootstrap p for 'indirect effect = 0': 2 x min(share of resamples <= 0, share >= 0)."""
    rng = np.random.default_rng(seed)
    n = len(iv)
    vals = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        i = rng.integers(0, n, n)
        X1 = np.column_stack([np.ones(n), iv[i]])
        a = np.linalg.lstsq(X1, m[i], rcond=None)[0][1]
        X2 = np.column_stack([np.ones(n), iv[i], m[i]])
        bcoef = np.linalg.lstsq(X2, dv[i], rcond=None)[0][2]
        vals[b] = a * bcoef
    p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))
    return float(min(1.0, max(p, 1.0 / n_bootstrap)))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def screen(df, iv_col="sweep_level", dv_col="asr", seed_col="seed", candidates=None, n_bootstrap=2000, seed=0, alpha=0.05):
    cands = [c for c in (candidates or DEFAULT_CANDIDATES) if c in df.columns]
    d = df.dropna(subset=[iv_col, dv_col, seed_col])
    seeds = sorted(d[seed_col].unique())
    half = len(seeds) // 2
    disc = d[d[seed_col].isin(seeds[:half])]
    rep = d[d[seed_col].isin(seeds[half:])]
    rows = []
    for c in cands:
        dd, rr = disc.dropna(subset=[c]), rep.dropna(subset=[c])
        if len(dd) < 12 or len(rr) < 12 or dd[c].std() == 0 or rr[c].std() == 0:
            rows.append({"candidate": c, "status": "skipped (too few rows or no variance)"})
            continue
        a = mediation_analysis(_z(dd[iv_col]), _z(dd[c]), _z(dd[dv_col]), n_bootstrap=n_bootstrap, seed=seed)
        p = _indirect_p(_z(dd[iv_col]), _z(dd[c]), _z(dd[dv_col]), n_bootstrap, seed)
        b = mediation_analysis(_z(rr[iv_col]), _z(rr[c]), _z(rr[dv_col]), n_bootstrap=n_bootstrap, seed=seed + 1)
        rows.append({"candidate": c, "status": "tested", "discovery_indirect": a["indirect_effect"],
                     "discovery_ci": [a["ci_low"], a["ci_high"]], "discovery_p": p,
                     "replication_indirect": b["indirect_effect"], "replication_ci": [b["ci_low"], b["ci_high"]],
                     "n_discovery": int(len(dd)), "n_replication": int(len(rr))})
    tested = [r for r in rows if r["status"] == "tested"]
    if tested:
        adj = holm([r["discovery_p"] for r in tested])
        for r, pa in zip(tested, adj):
            r["discovery_p_holm"] = float(pa)
            r["survives_discovery"] = bool(pa < alpha)
            same_sign = np.sign(r["discovery_indirect"]) == np.sign(r["replication_indirect"])
            ci_excl = not (r["replication_ci"][0] <= 0 <= r["replication_ci"][1])
            r["replicates"] = bool(r["survives_discovery"] and same_sign and ci_excl)
    return {"label": "EXPLORATORY: hypothesis-generating only; never a confirmatory verdict, never a substitute for M2",
            "k_candidates_tested": len(tested), "expected_false_hits_at_alpha": round(alpha * len(tested), 2),
            "discovery_seeds": [int(s) for s in seeds[:half]], "replication_seeds": [int(s) for s in seeds[half:]],
            "candidates": rows, "survivors": [r["candidate"] for r in tested if r.get("replicates")],
            "next_step": ("A survivor is a hypothesis for a NEW pre-registered run with fresh seeds (new experiment id). "
                          "No survivor means nothing here supports a different mediator.")}
