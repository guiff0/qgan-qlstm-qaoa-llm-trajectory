"""TOST equivalence tests: maths, wiring into the hypothesis script, and the verdict layer."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scripts.run_hypothesis_tests as hyp
from src.evaluation.statistical_tests import tost_correlation, tost_equivalence
from src.evaluation.verdicts import compute_verdicts


def test_equivalent_when_true_difference_is_tiny_and_n_is_large():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 1, 200), rng.normal(10.02, 1, 200)
    r = tost_equivalence(a, b, margin=0.5)
    assert r["equivalent"] and r["p_tost"] < 0.05
    assert -0.5 < r["ci_low"] < r["ci_high"] < 0.5          # 90% CI inside the margin <=> equivalent


def test_not_equivalent_when_difference_exceeds_margin():
    rng = np.random.default_rng(1)
    r = tost_equivalence(rng.normal(12, 1, 100), rng.normal(10, 1, 100), margin=0.5)
    assert not r["equivalent"] and r["p_tost"] > 0.5


def test_small_noisy_samples_cannot_show_equivalence_even_with_equal_means():
    """The point of TOST: 'p > .05' with n=3 is NOT evidence of equivalence."""
    a, b = np.array([9.0, 10.0, 11.0]), np.array([10.0, 11.0, 9.0])
    assert tost_equivalence(a, b, margin=0.5)["equivalent"] is False
    assert stats.ttest_ind(a, b, equal_var=False).pvalue > 0.9    # a plain t-test would "find no difference"


def test_ci_rule_matches_p_rule_and_is_symmetric():
    rng = np.random.default_rng(2)
    for shift in (0.0, 0.1, 0.3, 0.8):
        a, b = rng.normal(5 + shift, 1, 60), rng.normal(5, 1, 60)
        r, rs = tost_equivalence(a, b, 0.6), tost_equivalence(b, a, 0.6)
        assert r["equivalent"] == (-0.6 < r["ci_low"] and r["ci_high"] < 0.6)
        assert r["equivalent"] == rs["equivalent"] and np.isclose(r["p_tost"], rs["p_tost"])
        assert np.isclose(r["diff"], -rs["diff"])


def test_matches_reference_computation():
    a, b = np.array([1.0, 1.2, 0.9, 1.1, 1.05]), np.array([1.0, 0.95, 1.1, 1.0, 1.02, 0.98])
    m = 0.3
    r = tost_equivalence(a, b, m)
    # reference: shift one group by +/- margin and use scipy's one-sided Welch tests
    p_up = stats.ttest_ind(a, b + m, equal_var=False, alternative="less").pvalue      # H0 diff >= +m
    p_lo = stats.ttest_ind(a, b - m, equal_var=False, alternative="greater").pvalue   # H0 diff <= -m
    assert np.isclose(r["p_upper"], p_up) and np.isclose(r["p_lower"], p_lo)


def test_input_validation():
    with pytest.raises(ValueError):
        tost_equivalence([1.0], [1.0, 2.0], 0.5)
    with pytest.raises(ValueError):
        tost_equivalence([1.0, 2.0], [1.0, 2.0], 0.0)


def test_zero_variance_groups_do_not_divide_by_zero():
    assert tost_equivalence([1, 1, 1], [1, 1, 1], 0.1)["equivalent"]
    assert not tost_equivalence([2, 2, 2], [1, 1, 1], 0.1)["equivalent"]


def test_correlation_equivalence():
    assert tost_correlation(0.02, 400, r0=0.3)["equivalent"]
    assert not tost_correlation(0.02, 12, r0=0.3)["equivalent"]       # too few points to rule out |r| >= .3
    assert not tost_correlation(0.5, 400, r0=0.3)["equivalent"]


# ----------------------------------------------------------------------------- wiring
def _rows(slug, mt, n, fid, rng, sd=0.5):
    return [{"run_id": f"{slug}_20261031T12{k:02d}00Z", "model_type": mt, "seed": 43 + k,
             "fid": fid + rng.normal(0, sd), "noise_strength": 0.01} for k in range(n)]


def test_h1_reports_equivalence_next_to_the_superiority_test():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(_rows("qgan_llm", "QGAN-LLM", 12, 20.0, rng) + _rows("classical_gan_llm", "Classical GAN-LLM", 12, 20.1, rng))
    r = hyp.test_h1_synthetic_data_fidelity(df)
    eq = r["equivalence"]
    assert eq["metric"] == "fid" and eq["margin"] == pytest.approx(0.10 * abs(df[df.model_type == "Classical GAN-LLM"].fid.mean()))
    assert eq["equivalent"] is True
    v = compute_verdicts({"H1_synthetic_data_fidelity": r})["H1_synthetic_data_fidelity"]
    assert any("EQUIVALENT" in w for w in v["why"]) and v["equivalence"]["fid"]["equivalent"]


def test_h1_with_few_runs_has_no_equivalence_block():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(_rows("qgan_llm", "QGAN-LLM", 1, 20.0, rng) + _rows("classical_gan_llm", "Classical GAN-LLM", 1, 21.0, rng))
    assert hyp.test_h1_synthetic_data_fidelity(df).get("equivalence") is None


def test_h4_rmse_margin_is_relative_to_persistence():
    rng = np.random.default_rng(5)
    rows = []
    for mt, slug, mu in (("QGAN-LLM", "qgan_llm", 1.3e-4), ("Classical GAN-LLM", "classical_gan_llm", 1.3e-4)):
        for k in range(10):
            rows.append({"run_id": f"{slug}_20261031T12{k:02d}00Z", "model_type": mt, "seed": k,
                         "rmse": mu + rng.normal(0, 2e-7), "asr": 40 + rng.normal(0, 1),
                         "persistence_rmse": 1.64e-4})
    r = hyp.test_h4_quantum_advantage(pd.DataFrame(rows))
    assert r["rmse"]["equivalence"]["margin"] == pytest.approx(0.02 * 1.64e-4)
    assert r["rmse"]["equivalence"]["equivalent"] is True
    assert r["asr"]["equivalence"]["margin"] == 5.0


def test_margin_override_changes_the_decision(monkeypatch):
    rng = np.random.default_rng(6)
    a, b = rng.normal(10.5, 0.3, 40), rng.normal(10.0, 0.3, 40)
    monkeypatch.setitem(hyp.EQUIV_MARGINS, "asr", ("abs", 0.2))
    assert hyp._equivalence(a, b, "asr")["equivalent"] is False
    monkeypatch.setitem(hyp.EQUIV_MARGINS, "asr", ("abs", 2.0))
    assert hyp._equivalence(a, b, "asr")["equivalent"] is True


def test_correlation_results_carry_an_equivalence_block():
    rng = np.random.default_rng(7)
    x = rng.normal(size=60)
    df = pd.DataFrame({"entanglement_entropy": x, "asr": rng.normal(size=60)})
    res = hyp._entropy_correlation(df, "asr")
    assert "equivalence_r" in res and res["equivalence_r"]["r0"] == 0.3
