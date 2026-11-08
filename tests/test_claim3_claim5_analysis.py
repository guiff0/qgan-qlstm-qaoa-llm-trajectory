import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import run_hypothesis_tests as rht  # noqa: E402
from src.evaluation.decidability import detector_gate, poisoning_gate  # noqa: E402
from src.evaluation.paired_tests import paired_by_seed  # noqa: E402
from src.evaluation.verdicts import compute_verdicts  # noqa: E402

SEEDS = list(range(43, 53))


def grp(vals, seeds=SEEDS):
    return pd.DataFrame({"seed": seeds, "x": vals})


# ------------------------------------------------------------ paired by seed
def test_paired_removes_shared_seed_noise():
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 5, 10)                       # seed-level noise that both arms see
    a, b = grp(100 + shared - 1.0), grp(100 + shared + rng.normal(0, 0.1, 10))
    from scipy import stats
    welch_p = stats.ttest_ind(a["x"], b["x"], equal_var=False).pvalue
    res = paired_by_seed(a, b, "x")
    assert res["n_pairs"] == 10 and res["p_value"] < 0.001 < welch_p        # pairing finds what Welch cannot
    assert res["consistency_rule"]["met"] and res["sign_count"]["a_lower"] >= 8
    assert res["ci_low"] < res["mean_diff"] < res["ci_high"] and res["wilcoxon_p"] < 0.05


def test_paired_uses_only_shared_seeds_and_reports_the_rest():
    res = paired_by_seed(grp([1, 2, 3, 4, 5], seeds=[43, 44, 45, 46, 47]), grp([2, 3, 4], seeds=[44, 45, 99]), "x")
    assert res["seeds"] == [44, 45] or res["status"] == "insufficient_data"
    assert res["unpaired_seeds"]["a_only"] == [43, 46, 47] and res["unpaired_seeds"]["b_only"] == [99]


def test_paired_needs_three_pairs_and_the_columns():
    assert paired_by_seed(grp([1, 2], seeds=[43, 44]), grp([2, 3], seeds=[43, 44]), "x")["status"] == "insufficient_data"
    three = grp([1, 2, 3], seeds=[43, 44, 45])
    assert paired_by_seed(three, grp([2, 3, 4], seeds=[43, 44, 45]), "missing")["status"] == "insufficient_data"


def test_consistency_rule_8_of_10():
    d = [-1] * 7 + [1] * 3
    res = paired_by_seed(grp([float(v) for v in d]), grp([0.0] * 10), "x")
    assert res["sign_count"] == {"a_lower": 7, "a_higher": 3, "ties": 0} and not res["consistency_rule"]["met"]
    res = paired_by_seed(grp([float(v) for v in [-1] * 8 + [1] * 2]), grp([0.0] * 10), "x")
    assert res["consistency_rule"]["met"] and res["consistency_rule"]["needed"] == 8


def test_paired_equivalence_needs_a_small_enough_spread():
    rng = np.random.default_rng(1)
    tight = paired_by_seed(grp(rng.normal(0, 0.01, 10)), grp([0.0] * 10), "x", margin=0.1)
    wide = paired_by_seed(grp(rng.normal(0, 1.0, 10)), grp([0.0] * 10), "x", margin=0.1)
    assert tight["equivalence"]["equivalent"] and not wide["equivalence"]["equivalent"]


# --------------------------------------------------------------- decidability
def test_poisoning_floor_effect_is_not_decidable():
    g = poisoning_gate([6e-5, 7e-5, 1e-4, 4e-4], margin=5.0)
    assert g["decidable"] is False and "any model" in g["reason"] and g["remedy"]
    assert poisoning_gate([0.5, 3.0, 12.0, 8.0], margin=5.0)["decidable"] is True


def test_detector_at_chance_is_not_decidable_but_a_working_detector_is():
    rng = np.random.default_rng(2)
    at_chance = rng.binomial(204, 0.5, 25) / 204
    g = detector_gate(at_chance, 204)
    assert g["decidable"] is False and g["p_above_chance"] > 0.05 and "no signal" in g["reason"]
    assert detector_gate(rng.binomial(204, 0.7, 25) / 204, 204)["decidable"] is True
    assert detector_gate([np.nan], 204)["decidable"] is None


def _results(acc, poison):
    rows = []
    for exp_cond, slug, mt, noise in (("q", "qgan_llm", "QGAN-LLM", 0.01), ("nn", "qgan_llm_no_noise", "QGAN-LLM", 0.0),
                                      ("c", "classical_gan_llm", "Classical GAN-LLM", None)):
        for i, s in enumerate(SEEDS):
            rows.append({"run_id": f"{slug}_20261107T1{i:05d}Z", "model_type": mt, "seed": s,
                         "fid": 120 + 10 * (exp_cond == "q") + (s % 5), "rmse": 1e-4 + s * 1e-8,
                         "asr": 30 + (s % 7), "persistence_rmse": 1e-4, "forecast_path": "generator",
                         "poisoning_rmse_degradation_pct": poison[i], "noise_strength": noise,
                         "entanglement_entropy": 0.1 * i + (exp_cond == "q") * 0.05,
                         "llm_threat_accuracy": acc[i], "llm_threat_n": 204})
    return pd.DataFrame(rows)


def test_h3_and_h5_gates_turn_into_not_decidable_verdicts():
    rng = np.random.default_rng(3)
    df = _results(rng.binomial(204, 0.5, 10) / 204, np.abs(rng.normal(0, 1e-4, 10)))
    out = {"H3_poisoning_resistance": rht.test_h3_poisoning_resistance(df),
           "H5_threat_detection_tomography_correlation": rht.test_h5_tomography_correlation(df)}
    assert out["H3_poisoning_resistance"]["decidability"]["decidable"] is False
    assert out["H5_threat_detection_tomography_correlation"]["decidability"]["decidable"] is False
    v = compute_verdicts(out)
    for k in out:
        assert v[k]["verdict"] == "NOT_DECIDABLE" and v[k]["rule"] == "R9_not_decidable_by_design"
        assert v[k]["tier"] == "NONE" and v[k]["underlying"]["verdict"]
        assert "equivalence" not in v[k]                    # the vacuous TOST is not reported as a finding


def test_decidable_outcomes_keep_their_normal_verdicts_and_get_paired_evidence():
    rng = np.random.default_rng(4)
    df = _results(rng.binomial(204, 0.75, 10) / 204, np.abs(rng.normal(8, 3, 10)))
    out = {"H1_synthetic_data_fidelity": rht.test_h1_synthetic_data_fidelity(df),
           "H3_poisoning_resistance": rht.test_h3_poisoning_resistance(df),
           "H5_threat_detection_tomography_correlation": rht.test_h5_tomography_correlation(df)}
    assert out["H1_synthetic_data_fidelity"]["paired_by_seed"]["n_pairs"] == 10
    v = compute_verdicts(out)
    assert v["H3_poisoning_resistance"]["verdict"] != "NOT_DECIDABLE"
    assert v["H5_threat_detection_tomography_correlation"]["verdict"] != "NOT_DECIDABLE"
    assert any("paired by seed (fid)" in w for w in v["H1_synthetic_data_fidelity"]["why"])
    assert v["H1_synthetic_data_fidelity"]["paired_by_seed"]["fid"]["consistency_rule_met"] is True


def test_paired_is_off_when_experiments_are_pooled(tmp_path):
    rng = np.random.default_rng(5)
    a = _results(rng.binomial(204, 0.75, 10) / 204, np.abs(rng.normal(8, 3, 10))).assign(experiment_id="EXP-20261107-A-aaaa")
    b = _results(rng.binomial(204, 0.75, 10) / 204, np.abs(rng.normal(8, 3, 10))).assign(
        experiment_id="EXP-20261108-B-bbbb", run_id=lambda d: d["run_id"].str.replace("20261107", "20261108"))
    csv = tmp_path / "all.csv"
    pd.concat([a, b]).to_csv(csv, index=False)
    env = dict(os.environ, PYTHONPATH=ROOT)
    run = lambda *x: subprocess.run([sys.executable, "-m", "scripts.run_hypothesis_tests", "--results-csv", str(csv),
                                     "--out", str(tmp_path / "o.json"), *x], capture_output=True, text=True, cwd=ROOT, env=env)
    assert run().returncode != 0
    pooled = run("--experiment-id", "EXP-20261107-A-aaaa,EXP-20261108-B-bbbb", "--pool-experiments")
    assert pooled.returncode == 0, pooled.stderr[-500:]
    res = json.load(open(tmp_path / "o.json"))
    assert res["pooled_experiments"] and "paired_by_seed" not in res["H1_synthetic_data_fidelity"]
    assert res["seed_overlap_between_experiments"]
    single = run("--experiment-id", "EXP-20261107-A-aaaa")
    assert single.returncode == 0
    res = json.load(open(tmp_path / "o.json"))
    assert res["experiment_id"] == "EXP-20261107-A-aaaa" and "paired_by_seed" in res["H1_synthetic_data_fidelity"]


def test_frozen_margins_and_alpha_are_unchanged():
    """The additive analyses must not touch the pre-registered margins (results/preregistration_*.json)."""
    assert rht.EQUIV_MARGINS == {"fid": ("rel_ref", 0.10), "rmse": ("rel_persistence", 0.02), "asr": ("abs", 5.0),
                                 "poisoning": ("abs", 5.0), "r0": ("abs", 0.30)}
    assert rht.EQUIV_ALPHA == 0.05
