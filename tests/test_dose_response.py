import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.dose_response import h2_dose_verdict, jonckheere_terpstra, mediation_preconditions


def _g(means, n=10, sd=5, seed=0):
    r = np.random.default_rng(seed)
    return [r.normal(m, sd, n) for m in means]


def test_jt_detects_decreasing_trend_and_not_noise():
    assert jonckheere_terpstra(_g([60, 50, 40, 30, 20]), "decreasing")["p_one_sided"] < 0.001
    assert jonckheere_terpstra(_g([60, 50, 40, 30, 20]), "increasing")["p_one_sided"] > 0.99
    assert jonckheere_terpstra(_g([40] * 5, sd=15, seed=1), "decreasing")["p_one_sided"] > 0.05


def test_verdicts():
    assert h2_dose_verdict(_g([60, 50, 40, 30, 20]), 30.0, 10.0)["verdict"] == "SUPPORTED"
    assert h2_dose_verdict(_g([20, 30, 40, 50, 60]), 30.0, 10.0)["verdict"] == "CONTRADICTED"
    assert h2_dose_verdict(_g([40] * 5, n=40, sd=4, seed=2), 30.0, 10.0)["verdict"] == "EQUIVALENT"
    assert h2_dose_verdict(_g([40] * 5, sd=20, seed=3), 30.0, 10.0)["verdict"] == "INCONCLUSIVE"
    assert h2_dose_verdict(_g([60, 50, 40, 30, 20]), 1.0, 10.0)["verdict"] == "NOT_DECIDABLE"


def test_mediation_needs_a_total_effect():
    assert mediation_preconditions(0.91, True, 50)["ok"] is False
    assert mediation_preconditions(0.01, True, 50)["ok"] is True
    assert mediation_preconditions(0.01, False, 50)["problems"]
