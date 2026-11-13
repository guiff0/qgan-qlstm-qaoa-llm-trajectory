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
    for k, metric in (("H2_forecast_rmse", "RMSE"), ("H3_attack_success_qgan_vs_classical", "ASR")):
        r = results.get(k, {})
        single = r.get("status") == "exploratory_single_case"
        degenerate = r.get("status") == "degenerate_by_design"
        entry(k, f"Welch on run-level {metric}, paired by seed, row-level DM rule (RMSE)",
              "Crawford-Howell single case" if single else "degenerate by design" if degenerate else "Welch two-sample t",
              "one quantum run: single-case test, exploratory" if single else
              "generator not on the forecast path (head_only rows)" if degenerate else "")
    h4 = results.get("H4_poisoning_resistance_noise", {})
    why4 = ("decidability gate failed: " + h4["decidability"]["gate"]
            if isinstance(h4.get("decidability"), dict) and h4["decidability"].get("decidable") is False else "")
    entry("H4_poisoning_resistance_noise", "Welch no_noise vs noisy on matched configs + TOST",
          h4.get("design", h4.get("status", "not run")), why4)
    h5 = results.get("H5_noise_attack_success", {})
    entry("H5_noise_attack_success", "Welch no_noise vs noisy ASR on matched configs + TOST",
          h5.get("design", h5.get("status", "not run")),
          "matched pair missing: untestable" if h5.get("design") == "matched_pair_missing" else "")
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
