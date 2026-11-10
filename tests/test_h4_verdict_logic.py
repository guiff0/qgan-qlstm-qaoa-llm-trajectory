"""Synthetic cases for the H4 verdict path. Pure python: no torch, no PennyLane, no run data.

Scope: only the n-per-group rule is a bug fix against the frozen plan (config/analysis_plan.yaml). Nothing here tests a
new decision rule, and none of it is validated against Experiment 1 results.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.verdicts import MIN_N_CONFIRMATORY_PER_GROUP, compute_verdicts, verdict_h4  # noqa: E402

PLAN = os.path.join(os.path.dirname(__file__), "..", "config", "analysis_plan.yaml")


def _metric(t, p, nq, nc):
    return {"status": "ok", "t_statistic": t, "p_value": p, "n_per_group": {"QGAN-LLM": nq, "Classical GAN-LLM": nc}}


def test_confirmatory_threshold_matches_the_frozen_plan():
    with open(PLAN, encoding="utf-8") as fh:
        h = yaml.safe_load(fh)["hypotheses"]
    names = {"H1": "H1_synthetic_data_fidelity", "H3": "H3_poisoning_resistance", "H4": "H4_quantum_advantage"}
    for short, full in names.items():
        assert h[full]["replication"]["minimum_for_confirmatory"] == MIN_N_CONFIRMATORY_PER_GROUP[short]


def test_h4_three_vs_nine_runs_is_exploratory_not_confirmatory():
    res = {"rmse": _metric(-1.2, 0.26, 3, 9), "asr": _metric(+3.5, 0.005, 3, 9)}   # ASR significant, opposite direction
    v = verdict_h4(res)
    assert v["tier"] == "EXPLORATORY"
    assert v["verdict"] not in ("SUPPORTED", "CONTRADICTED")
    # CURRENT ranking in verdict_h4 does not order SUGGESTIVE_* verdicts, so the non-significant RMSE sub-verdict
    # (INCONCLUSIVE) is the headline and the significant exploratory ASR is not surfaced there. Pinned so a later
    # amendment that changes the ranking has to change this test on purpose.
    assert v["verdict"] == "INCONCLUSIVE" and v["rule"] == "R4_underpowered_not_significant"


def test_h4_significant_asr_alone_at_three_vs_nine_is_suggestive_against_not_contradicted():
    v = verdict_h4({"asr": _metric(+3.5, 0.005, 3, 9)})
    assert v["tier"] == "EXPLORATORY" and v["verdict"] == "SUGGESTIVE_AGAINST"
    assert any("analysis plan requires 10" in b for b in v["blockers"])


def test_h4_same_statistics_with_ten_per_group_are_confirmatory():
    res = {"rmse": _metric(-1.2, 0.26, 10, 10), "asr": _metric(+3.5, 0.005, 10, 10)}
    v = verdict_h4(res)
    assert v["tier"] == "CONFIRMATORY" and v["verdict"] == "CONTRADICTED"


def test_h4_ten_vs_nine_is_still_exploratory_because_the_smaller_arm_governs():
    v = verdict_h4({"rmse": _metric(-3.0, 0.004, 10, 9)})
    assert v["tier"] == "EXPLORATORY" and v["verdict"] == "SUGGESTIVE_FOR"


def test_h4_ten_each_supported_when_both_expected_direction():
    res = {"rmse": _metric(-3.0, 0.004, 10, 10), "asr": _metric(-2.5, 0.01, 10, 10)}
    v = verdict_h4(res)
    assert v["tier"] == "CONFIRMATORY" and v["verdict"] == "SUPPORTED"


def test_h4_not_significant_at_low_n_is_inconclusive_underpowered_rule():
    res = {"rmse": _metric(-1.0, 0.30, 3, 9), "asr": _metric(0.5, 0.6, 3, 9)}
    v = verdict_h4(res)
    assert v["verdict"] == "INCONCLUSIVE" and v["rule"] == "R4_underpowered_not_significant"


def test_exploratory_h4_does_not_enter_the_holm_family():
    out = {"H4_quantum_advantage": {"rmse": _metric(-1.2, 0.26, 3, 9), "asr": _metric(+3.5, 0.005, 3, 9)}}
    v = compute_verdicts(out)["H4_quantum_advantage"]
    assert v["tier"] == "EXPLORATORY" and "p_holm" not in v
