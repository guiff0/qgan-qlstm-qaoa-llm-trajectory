"""
Deterministic, explained verdicts for the H1-H5 / M2 hypothesis tests.

WHY THIS EXISTS
    run_hypothesis_tests.py's old "status" only said whether a test could RUN
    (ok / exploratory_single_case / degenerate_by_design ...). "ok" therefore
    meant "a p-value was computed", not "the hypothesis holds" -- H5 was "ok"
    with p = 0.38 and CI [-0.84, 0.99], and M2 was "ok" with a mediation CI
    containing 0.  This module keeps that execution status and adds two more
    independent questions, answered by fixed rules (no randomness, no LLM, no
    judgement calls at run time):

      VERDICT   what the test says about the hypothesis, in its stated direction
                  SUPPORTED     significant, expected direction, CONFIRMATORY evidence
                  CONTRADICTED  significant, OPPOSITE direction, CONFIRMATORY evidence
                  SUGGESTIVE_FOR / SUGGESTIVE_AGAINST
                                significant, but only EXPLORATORY evidence (do not cite as a result)
                  NOT_SUPPORTED adequately sized test, no significant effect
                  INCONCLUSIVE  not significant and the test is too small/weak
                                to call it a null (a "lean" is still reported)
                  UNTESTABLE    the design cannot produce an answer (see blockers)
      TIER      how far the verdict can be trusted
                  CONFIRMATORY  enough independent observations, no known confound
                  EXPLORATORY   a p-value exists but n is small, rows are not
                                independent runs, or conditions are confounded
                  NONE          no usable test

    Every verdict carries the rule id that produced it, so the same input
    always yields the same output and every conclusion can be traced to a line
    below.

ASSUMPTIONS YOU MUST CONFIRM AGAINST Ch.3 (edit EXPECTED below if they differ):
    H1  QGAN-LLM has LOWER Frechet distance than Classical GAN-LLM (lower = better)
    H2  more entanglement entropy -> LOWER attack success rate  (r < 0)
    H3  no-noise poisoning degradation > noisy degradation       (t > 0)
    H4  QGAN-LLM has LOWER RMSE / ASR than Classical GAN-LLM
    H5  more entanglement entropy -> HIGHER threat-detection accuracy (r > 0)
    M2  indirect effect through entanglement is non-zero (bootstrap CI excludes 0)
"""
from __future__ import annotations

import math

ALPHA = 0.05
MIN_N_PER_GROUP = 3          # two-sample tests: independent runs per condition
MIN_N_CORRELATION = 10       # correlations: points (configs/runs)
EXPECTED = {"H1": -1, "H2": -1, "H3": +1, "H4": -1, "H5": +1}

NAMES = {
    "H1_synthetic_data_fidelity": "H1",
    "H2_adversarial_resilience": "H2",
    "H3_poisoning_resistance": "H3",
    "H4_quantum_advantage": "H4",
    "H5_threat_detection_tomography_correlation": "H5",
    "M2_DV2_mediation": "M2",
}


# ----------------------------------------------------------------- helpers
def _sign(x) -> int:
    return 0 if x is None or x == 0 or (isinstance(x, float) and math.isnan(x)) else (1 if x > 0 else -1)


def _fmt_p(p, with_op: bool = False) -> str:
    if p is None:
        return "n/a"
    txt = "< 0.001" if p < 0.001 else f"{p:.3f}"
    return txt if (with_op and p < 0.001) else (f"= {txt}" if with_op else txt.replace("< ", "<"))


def _n_for_correlation(r: float, power_z: float = 0.8416, alpha_z: float = 1.96) -> int | None:
    """Sample size for 80 % power, two-sided alpha 0.05 (Fisher z)."""
    if r is None or abs(r) >= 1 or abs(r) < 1e-6:
        return None
    return int(math.ceil(((alpha_z + power_z) / math.atanh(abs(r))) ** 2 + 3))


def _out(verdict, tier, rule, headline, why=(), blockers=(), next_action="", lean=None, **extra):
    d = {"verdict": verdict, "tier": tier, "rule": rule, "lean": lean, "headline": headline,
         "why": list(why), "blockers": list(blockers), "next_action": next_action}
    d.update(extra)
    return d


def _decide(p, observed_sign, expected_sign, tier, label, alpha=ALPHA):
    """Core rule table. Returns (verdict, rule_id, lean).
    A significant result is only SUPPORTED / CONTRADICTED when the evidence tier is CONFIRMATORY;
    on EXPLORATORY evidence it is reported as SUGGESTIVE_FOR / SUGGESTIVE_AGAINST."""
    lean = ("for" if observed_sign == expected_sign else "against") if observed_sign else None
    if p is None:
        return "INCONCLUSIVE", "R0_no_p_value", lean
    if p < alpha and observed_sign:
        agree = observed_sign == expected_sign
        if tier == "CONFIRMATORY":
            return (("SUPPORTED", "R1_sig_expected_direction", "for") if agree
                    else ("CONTRADICTED", "R2_sig_opposite_direction", "against"))
        return (("SUGGESTIVE_FOR", "R1x_sig_expected_direction_exploratory", "for") if agree
                else ("SUGGESTIVE_AGAINST", "R2x_sig_opposite_direction_exploratory", "against"))
    if tier == "CONFIRMATORY":
        return "NOT_SUPPORTED", "R3_adequate_n_not_significant", lean
    return "INCONCLUSIVE", "R4_underpowered_not_significant", lean


def _holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


# ------------------------------------------------------------ per-hypothesis
def _two_group(res, key, a_name, b_name, exp_sign, confound=None, hyp_label=""):
    """res is a _compare_two() result where group a is `a_name`, b is `b_name`.
    exp_sign is the expected sign of (a - b)."""
    st = res.get("status")
    if st == "insufficient_data":
        return _out("UNTESTABLE", "NONE", "U1_insufficient_data",
                    f"{key}: not enough runs to test.", [res.get("message", "")],
                    ["too few runs in at least one condition"], res.get("how_to_fix", ""))
    if st == "error" or "error" in res:
        return _out("UNTESTABLE", "NONE", "U2_test_error", f"{key}: test failed to run.",
                    [res.get("error", "")], ["fix the error, then re-run"], "")
    t, p, n = res.get("t_statistic"), res.get("p_value"), res.get("n_per_group", {})
    single = st == "exploratory_single_case"
    # sign of (a - b)
    s = _sign(t)
    if single and res.get("single_condition") != a_name:
        s = -s
    blockers, why = [], []
    if single:
        blockers.append(f"{res['single_condition']} has only 1 run (two-sample test impossible)")
        mean_s, case = res.get("control_mean"), res.get("case_value")
        why.append(f"single {res['single_condition']} run = {case:.4g}; {res['sample_condition']} "
                   f"mean = {mean_s:.4g} (sd {res['control_sd']:.3g}, n={res['n_controls']})")
    else:
        mn = min(n.values()) if n else 0
        if mn < MIN_N_PER_GROUP:
            blockers.append(f"fewer than {MIN_N_PER_GROUP} independent runs in a condition ({n})")
    if confound:
        blockers.append(confound)
    tier = "EXPLORATORY" if blockers else "CONFIRMATORY"
    verdict, rule, lean = _decide(p, s, exp_sign, tier, key)
    direction = {+1: f"{a_name} > {b_name}", -1: f"{a_name} < {b_name}", 0: "no difference"}[s]
    expected = {+1: f"{a_name} > {b_name}", -1: f"{a_name} < {b_name}"}[exp_sign]
    why.insert(0, f"observed direction: {direction}; hypothesis expects: {expected}")
    why.append(f"p {_fmt_p(p, True)}, alpha = {ALPHA}")
    return _out(verdict, tier, rule,
                f"{hyp_label}: {verdict} ({tier.lower()}) -- observed {direction}, p={_fmt_p(p)}.",
                why, blockers, res.get("how_to_fix", ""), lean, p_value=p)


def verdict_h1(res):
    conf = ("QGAN-LLM runs pooled across different configs (qubits/topology/noise), so they are not "
            "repeats of one condition" if res.get("pooled_qgan_configs") else None)
    if "f_statistic" in res:       # ANCOVA path: no direction reported
        p = res.get("p_value")
        return _out("INCONCLUSIVE" if p is None or p >= ALPHA else "SUPPORTED",
                    "CONFIRMATORY", "R5_ancova",
                    f"H1: ANCOVA group effect p={_fmt_p(p)} (direction not reported -- inspect group means).",
                    [f"F = {res['f_statistic']:.3g}, partial eta^2 = {res['partial_eta_squared']:.3g}"],
                    ["ANCOVA output has no sign: check group means before claiming direction"], "", p_value=p)
    return _two_group(res, "H1", "QGAN-LLM", "Classical GAN-LLM", EXPECTED["H1"], conf, "H1 (QGAN FID lower)")


def verdict_h3(res):
    return _two_group(res, "H3", "no_noise", "noisy", EXPECTED["H3"], res.get("confound_warning"),
                      "H3 (noise lowers poisoning degradation)")


def _corr(res, hyp, expected, label, extra_blockers=()):
    if "error" in res and "r" not in res:
        return _out("UNTESTABLE", "NONE", "U2_test_error", f"{hyp}: {res['error']}", [res["error"]], [], "")
    if res.get("status") == "insufficient_data":
        return _out("UNTESTABLE", "NONE", "U1_insufficient_data", f"{hyp}: too few points to correlate.",
                    [res.get("message", "")], ["need >= 4 points just to compute r"], "")
    r, p, df = res["r"], res["p_value"], res["df"]
    n = df + 2
    lo, hi = res["ci_low"], res["ci_high"]
    blockers = list(extra_blockers)
    if n < MIN_N_CORRELATION:
        blockers.append(f"only n={n} points (< {MIN_N_CORRELATION}); each point is one model config, not an independent run")
    tier = "EXPLORATORY" if blockers else "CONFIRMATORY"
    verdict, rule, lean = _decide(p, _sign(r), expected, tier, hyp)
    ci_zero = lo <= 0 <= hi
    why = [f"r = {r:.3f}, 95% CI [{lo:.2f}, {hi:.2f}], p {_fmt_p(p, True)}, n = {n}",
           f"hypothesis expects r {'>' if expected > 0 else '<'} 0; observed r {'>' if r > 0 else '<'} 0"]
    if ci_zero:
        why.append("CI contains 0: the data are compatible with no relationship at all")
    nn_obs, nn_half = _n_for_correlation(r), _n_for_correlation(0.5)
    if verdict == "INCONCLUSIVE":
        why.append(f"~{nn_obs} points would give 80% power if the true r equals the observed {abs(r):.2f}; "
                   f"{nn_half} points if the true r is 0.5")
    if abs(r) > 0.99 and n < MIN_N_CORRELATION:
        why.append("|r| > 0.99 on so few points usually reflects configs differing in many ways at once, "
                   "not a clean dose-response")
    return _out(verdict, tier, rule, f"{label}: {verdict} ({tier.lower()}) -- r={r:.2f}, p={_fmt_p(p)}, n={n}.",
                why, blockers,
                f"Collect >= {max(MIN_N_CORRELATION, nn_half or 0)} independent runs spanning entanglement levels "
                f"(vary seed AND entanglement setting)." if blockers else "", lean, p_value=p, r=r)


def verdict_h2(res):
    return _corr(res, "H2", EXPECTED["H2"], "H2 (entanglement vs ASR)")


def verdict_h5(res):
    det = res.get("detector", "")
    extra = ["detector is the interim classical RandomForest, not the LLM detector"] if "classical" in det else ()
    return _corr(res, "H5", EXPECTED["H5"], "H5 (entanglement vs detection accuracy)", extra)


def verdict_h4(res):
    parts = {k: res.get(k, {}) for k in ("rmse", "asr")}
    sts = {k: v.get("status") for k, v in parts.items()}
    if all(s == "degenerate_by_design" for s in sts.values()):
        return _out("UNTESTABLE", "NONE", "U3_degenerate_by_design",
                    "H4 (quantum advantage): UNTESTABLE -- RMSE and ASR are identical for QGAN and classical GAN.",
                    [parts["rmse"].get("reason", ""), parts["asr"].get("reason", ""),
                     "forecast head is a closed-form linear fit on REAL rows; the generator never reaches the forecast"],
                    ["generator is not on the forecast path -- more seeds cannot change this"],
                    "Feed generator output into the forecast (e.g. train the head on real + synthetic rows) "
                    "or restate H4 as a claim about generation quality (H1) only.")
    sub = [verdict_h4_metric(k, v) for k, v in parts.items() if v.get("status") not in (None, "degenerate_by_design")]
    if not sub:
        return _out("UNTESTABLE", "NONE", "U2_test_error", "H4: no testable metric.", [], [], "")
    order = {"CONTRADICTED": 0, "SUPPORTED": 1, "INCONCLUSIVE": 2, "NOT_SUPPORTED": 3}
    return sorted(sub, key=lambda d: order.get(d["verdict"], 9))[0]


def verdict_h4_metric(k, v):
    return _two_group(v, f"H4/{k}", "QGAN-LLM", "Classical GAN-LLM", EXPECTED["H4"], None, f"H4 ({k.upper()})")


def verdict_m2(res):
    if "error" in res:
        return _out("UNTESTABLE", "NONE", "U2_test_error", f"M2: {res['error']}", [res["error"]], [], "")
    lo, hi, ie = res["ci_low"], res["ci_high"], res["indirect_effect"]
    blockers = [f"{res.get('n_observations')} rows are model CONFIGS, not independent runs (unit-of-analysis caveat)"]
    excl = not (lo <= 0 <= hi)
    why = [f"indirect effect = {ie:.3f}, bootstrap 95% CI [{lo:.3f}, {hi:.3f}] ({res.get('n_bootstrap')} resamples)",
           f"proportion mediated = {res.get('proportion_mediated', float('nan')):.0%}",
           f"a-path = {res['path_a']:.3f}, b-path = {res['path_b']:.3f}, total c = {res['path_c']:.3f}, direct c' = {res['path_c_prime']:.3f}"]
    if excl:
        v, rule = "SUGGESTIVE_FOR", "R6x_indirect_ci_excludes_0_exploratory"
    else:
        v, rule = "INCONCLUSIVE", "R7_indirect_ci_includes_0"
        why.append("CI contains 0: no evidence that entanglement mediates the effect")
    return _out(v, "EXPLORATORY", rule, f"M2 (entanglement mediates): {v} (exploratory) -- indirect CI "
                f"[{lo:.2f}, {hi:.2f}] {'excludes' if excl else 'includes'} 0.", why, blockers,
                "Need repeated independent runs per config before mediation can be confirmatory.",
                "for" if excl else None)


_FN = {"H1": verdict_h1, "H2": verdict_h2, "H3": verdict_h3, "H4": verdict_h4, "H5": verdict_h5, "M2": verdict_m2}


def compute_verdicts(out: dict) -> dict:
    """Pure function: results dict (as written by run_hypothesis_tests) -> verdicts dict."""
    v = {}
    for key, short in NAMES.items():
        try:
            v[key] = _FN[short](out.get(key, {}))
        except Exception as exc:  # never let the verdict layer break the run
            v[key] = _out("UNTESTABLE", "NONE", "U4_verdict_error", f"{short}: verdict failed ({exc})",
                          [repr(exc)], [], "")
    # Holm correction over the confirmatory family only
    fam = {k: d["p_value"] for k, d in v.items() if d["tier"] == "CONFIRMATORY" and d.get("p_value") is not None}
    for k, adj in _holm(fam).items():
        v[k]["p_holm"] = adj
        if v[k]["verdict"] in ("SUPPORTED", "CONTRADICTED") and adj >= ALPHA:
            v[k].update(verdict="INCONCLUSIVE", rule="R8_fails_holm_correction")
            v[k]["why"].append(f"Holm-adjusted p = {adj:.3f} >= {ALPHA}")
    return v


def format_table(verdicts: dict, detail: bool = True) -> str:
    rows = [("HYPOTHESIS", "VERDICT", "EVIDENCE", "RULE")]
    for k, d in verdicts.items():
        rows.append((k, d["verdict"] + (f" (leans {d['lean']})" if d["verdict"] == "INCONCLUSIVE" and d.get("lean") else ""),
                     d["tier"], d["rule"]))
    w = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    lines.insert(1, "  ".join("-" * x for x in w))
    if detail:
        for k, d in verdicts.items():
            lines += ["", d["headline"]]
            lines += [f"    why:      {x}" for x in d["why"] if x]
            lines += [f"    blocker:  {x}" for x in d["blockers"]]
            if d["next_action"]:
                lines.append(f"    next:     {d['next_action']}")
    return "\n".join(lines)
