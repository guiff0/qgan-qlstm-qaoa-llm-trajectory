"""Reviewer round 5 (Ch.4): one analysed design, interim != confirmatory, H3 untestable without a matched run,
H5 needs >= 30 runs, one controlling label per hypothesis, no silent count mismatches."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import run_hypothesis_tests as rht
from src.evaluation import verdicts as V
from src.evaluation.study_status import study_status


def _rows(slug, n, forecast_path="generator", **extra):
    return [dict(run_id=f"{slug}_2026110{i}T000000Z", model_type="QGAN-LLM" if "qgan" in slug else "Classical GAN-LLM",
                 forecast_path=forecast_path, **extra) for i in range(n)]


def test_only_generator_on_path_rows_are_analysed():
    df = pd.DataFrame(_rows("qgan_llm", 2) + _rows("qgan_llm_no_noise", 1, forecast_path="head_only"))
    kept, info = rht.authoritative_design(df)
    assert len(kept) == 2 and info["excluded_pilot_rows"] == {"qgan_llm_no_noise": 1}
    assert len(rht.authoritative_design(df, include_pilot=True)[0]) == 3


def test_h3_is_untestable_without_a_matched_no_noise_run():
    df = pd.DataFrame(_rows("qgan_llm", 2, noise_strength=0.01, poisoning_rmse_degradation_pct=3.0))
    res = rht.test_h3_poisoning_resistance(df)
    assert res["status"] == "insufficient_data" and "matched no-noise" in res["untestable_reason"]
    v = V.compute_verdicts({"H4_poisoning_resistance_noise": res})["H4_poisoning_resistance_noise"]
    assert v["verdict"] == "UNTESTABLE" and v["chapter_label"] == "Untestable"


def test_interim_sample_is_never_confirmatory():
    st = study_status({"qgan_llm": 3, "classical_gan_llm": 10})
    assert not st["complete"] and st["planned_runs"] == 40
    d = {"verdict": "SUPPORTED", "tier": "CONFIRMATORY", "rule": "R1", "headline": "H1: SUPPORTED (confirmatory)",
         "blockers": [], "why": [], "lean": "for"}
    V.apply_interim_gate(d, st)
    assert d["tier"] == "EXPLORATORY" and d["verdict"] == "SUGGESTIVE_FOR" and d["interim"]
    full = study_status({s: 10 for s in __import__("src.evaluation.study_status", fromlist=["x"]).preregistered_slugs()})
    assert full["complete"]


def test_h5_needs_thirty_independent_runs():
    res = dict(r=0.3, p_value=0.001, df=15, ci_low=0.1, ci_high=0.5, unit="run", min_runs_per_level=5,
               manipulation_check_level_vs_entropy=dict(p_value=0.01, spearman_rho=0.5))
    d = V._corr(res, "H5", +1, "H5", min_n=V.MIN_N_H5)          # n = 17
    assert d["tier"] == "EXPLORATORY" and any("< 30" in b for b in d["blockers"])
    assert d["verdict"] == "SUGGESTIVE_FOR"


def test_chapter_labels_are_single_vocabulary():
    mk = lambda v, tier="CONFIRMATORY", lean=None: {"verdict": v, "tier": tier, "lean": lean}
    assert V.chapter_label(mk("INCONCLUSIVE", "EXPLORATORY", "against")).startswith("Inconclusive (exploratory)")
    assert V.chapter_label(mk("NOT_SUPPORTED")) == "Not supported"
    assert V.chapter_label(mk("NOT_DECIDABLE", "NONE")).startswith("Untestable")


def test_count_audit_flags_a_mismatch():
    out = {"n_runs_per_config": {"qgan_llm": 1, "classical_gan_llm": 4},
           "H1_synthetic_data_fidelity": {"n_per_group_used": {"QGAN-LLM": 2, "Classical GAN-LLM": 5}}}
    a = rht.count_audit(out)
    assert not a["consistent"] and len(a["issues"]) == 2
