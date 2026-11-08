"""Machine-readable analysis plan (config/analysis_plan.yaml) and the check that it matches the code.

Reviewer claim 5: define the experimental unit before choosing each analysis, and for each research question state the
independent variable and levels, the outcome and scale, the independent unit and replication, the planned contrast and
assumptions, and a fallback. The YAML holds that; this module loads it, validates that nothing is missing, renders the
table for Chapter 3, and records which planned analysis actually ran (and whether a fallback fired) in every
hypothesis_test_results.json.
"""
from __future__ import annotations

import hashlib
import os

import yaml

PLAN_PATH = os.path.join("config", "analysis_plan.yaml")
REQUIRED = ("question", "independent_variable", "outcome", "unit", "replication", "contrast", "assumptions", "fallback")


def load_plan(path: str = PLAN_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def plan_sha(path: str = PLAN_PATH) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def validate(plan: dict, hypotheses) -> list[str]:
    """Problems found (empty list = the plan is complete for every hypothesis the code tests)."""
    problems = []
    if not plan.get("unit_of_analysis"):
        problems.append("unit_of_analysis missing")
    hyp = plan.get("hypotheses") or {}
    for h in hypotheses:
        if h not in hyp:
            problems.append(f"{h}: not in the analysis plan")
            continue
        for k in REQUIRED:
            if not hyp[h].get(k):
                problems.append(f"{h}: '{k}' is empty")
        rep = hyp[h].get("replication") or {}
        for k in ("planned_per_condition", "minimum_for_confirmatory", "minimum_for_any_test"):
            if not isinstance(rep.get(k), int):
                problems.append(f"{h}: replication.{k} must be an integer")
    for h in hyp:
        if h not in hypotheses:
            problems.append(f"{h}: in the plan but not tested by the code")
    return problems


def planned_vs_ran(results: dict) -> dict:
    """For each hypothesis: the planned primary analysis, what ran, and whether a fallback fired."""
    out = {}

    def entry(key, planned, ran, fallback):
        out[key] = {"planned": planned, "ran": ran, "fallback_triggered": bool(fallback), "why": fallback or ""}

    h = results.get("H1_synthetic_data_fidelity", {})
    entry("H1_synthetic_data_fidelity", "Welch on run-level FID (+ ANCOVA when a VIX covariate exists)",
          "ANCOVA" if "f_statistic" in h else "Welch two-sample t" if h.get("status") == "ok" else h.get("status", "not run"),
          h.get("note_covariate") or (h.get("message") if h.get("status") not in (None, "ok") else ""))
    for k in ("H2_adversarial_resilience", "H5_threat_detection_tomography_correlation", "M2_DV2_mediation"):
        r = results.get(k, {})
        mc = r.get("manipulation_check_level_vs_entropy") or {}
        why = ""
        if r.get("design") != "controlled_sweep":
            why = "no controlled sweep in the data: configs pooled (exploratory)"
        elif mc and (mc.get("p_value", 1) >= 0.05 or mc.get("spearman_rho", 0) <= 0):
            why = "manipulation check failed"
        if isinstance(r.get("decidability"), dict) and r["decidability"].get("decidable") is False:
            why = (why + "; " if why else "") + "decidability gate failed: " + r["decidability"]["gate"]
        entry(k, "run-level correlation / bootstrap mediation on the controlled sweep",
              r.get("design", "not run"), why)
    h3 = results.get("H3_poisoning_resistance", {})
    why3 = ("decidability gate failed: " + h3["decidability"]["gate"]
            if isinstance(h3.get("decidability"), dict) and h3["decidability"].get("decidable") is False else "")
    entry("H3_poisoning_resistance", "Welch no_noise vs noisy on matched configs + TOST",
          h3.get("design", h3.get("status", "not run")), why3)
    h4 = results.get("H4_quantum_advantage", {})
    single = any(isinstance(h4.get(m), dict) and h4[m].get("status") == "exploratory_single_case" for m in ("rmse", "asr"))
    entry("H4_quantum_advantage", "Welch on run-level RMSE/ASR, paired by seed, row-level DM rule",
          "Crawford-Howell single case" if single else "Welch two-sample t",
          "one quantum run: single-case test, exploratory" if single else "")
    return out


def render_markdown(plan: dict) -> str:
    """A pipe table for the manuscript (one row per hypothesis)."""
    rows = ["| Hypothesis | Independent variable (levels) | Outcome (scale) | Unit and replication | Planned contrast | Fallback |",
            "|---|---|---|---|---|---|"]
    for key, h in plan["hypotheses"].items():
        iv, oc, rep = h["independent_variable"], h["outcome"], h["replication"]
        levels = ", ".join(str(x) for x in iv.get("levels", []))
        rows.append(f"| {key.split('_')[0]} | {iv['name']} ({levels}) | {oc['name']} ({oc['scale']}) | "
                    f"{h['unit']}; {rep['planned_per_condition']} per condition, >= {rep['minimum_for_confirmatory']} "
                    f"for a confirmatory call | {h['contrast']} | {h['fallback']} |")
    return "\n".join(rows)
