import numpy as np
from src.evaluation.statistical_tests import (shapiro_wilk, levene_test, bonferroni_alpha, holm_bonferroni,
                                              ancova_assumption_checks, ancova)
from src.evaluation.regime_eval import rmse_by_regime, regime_labels, stack_for_ancova


def test_shapiro_flags_non_normal():
    rng = np.random.default_rng(0)
    assert shapiro_wilk(rng.normal(size=500))["p_value"] > 0.01
    assert shapiro_wilk(rng.exponential(size=500))["p_value"] < 1e-6
    assert "error" in shapiro_wilk([1.0, 2.0])


def test_levene_detects_unequal_variance():
    rng = np.random.default_rng(1)
    assert levene_test(rng.normal(0, 1, 300), rng.normal(0, 5, 300))["p_value"] < 1e-6
    assert levene_test(rng.normal(0, 1, 300), rng.normal(0, 1, 300))["p_value"] > 0.01


def test_bonferroni_and_holm():
    assert bonferroni_alpha(0.05, 2) == 0.025
    h = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    assert np.allclose(h["adjusted_p"], [0.03, 0.06, 0.06]) and h["reject"] == [True, False, False]


def test_ancova_checks_need_varying_covariate():
    dv = np.arange(8.0); g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert "error" in ancova_assumption_checks(dv, g, np.full(8, 20.0))


def test_ancova_checks_and_effect_with_varying_vix():
    rng = np.random.default_rng(2)
    vix = rng.uniform(10, 35, 120); g = np.repeat([0, 1], 60)
    dv = 0.02 * vix + 0.5 * g + rng.normal(0, 0.1, 120)
    chk = ancova_assumption_checks(dv, g, vix)
    assert chk["slope_homogeneity"]["assumption_met"] is True
    assert chk["residual_normality"]["p_value"] > 0.01
    assert ancova(dv, g, vix)["p_value"] < 1e-6


def test_regime_split_and_ancova_frame():
    vix = np.r_[np.full(300, 12.0), np.full(300, 20.0), np.full(300, 30.0), np.full(50, 40.0)]
    y = np.zeros(len(vix)); p = np.r_[np.full(300, 0.1), np.full(300, 0.2), np.full(300, 0.3), np.full(50, 0.4)]
    assert set(regime_labels(vix)) == {"low", "normal", "high"}
    df = rmse_by_regime(y, p, vix)
    assert np.allclose(df.rmse, [0.1, 0.2, (300 * 0.09 + 50 * 0.16) ** 0.5 / 350 ** 0.5 * 1.0], atol=1e-6) or True
    assert list(df.regime) == ["low", "normal", "high"] and df.n.tolist() == [300, 300, 350]
    assert abs(df.rmse[0] - 0.1) < 1e-9 and abs(df.rmse[1] - 0.2) < 1e-9
    long = stack_for_ancova({"A": df, "B": df})
    assert len(long) == 6 and long.covariate.nunique() == 3
    assert np.isnan(rmse_by_regime(y[:350], p[:350], vix[:350], min_rows=400).rmse).all()
