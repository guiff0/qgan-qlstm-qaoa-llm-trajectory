"""The selector comparison is exploratory, separate from the three-arm methodology, and its bookkeeping is exact."""
import ast
import os

import numpy as np
import pytest

pytest.importorskip("torch")
from scripts import qaoa_selector_comparison as sc          # noqa: E402
from scripts.qaoa_solution_audit import all_qubo_values     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _q():
    Q = np.array([[-1.0, 2.0, 0.0], [0.0, -3.0, 0.5], [0.0, 0.0, 1.0]])
    return {"cand": np.array([7, 4, 9]), "Q": Q, "const": 0.25, "values": all_qubo_values(Q, 0.25), "k": 2}


def test_all_qubo_values_matches_a_direct_evaluation_for_every_bitstring():
    q = _q()
    for idx in range(8):
        x = np.array([(idx >> (2 - w)) & 1 for w in range(3)], dtype=float)
        direct = x @ np.diag(q["Q"]) + sum(q["Q"][i, j] * x[i] * x[j] for i in range(3) for j in range(i + 1, 3)) + 0.25
        assert q["values"][idx] == pytest.approx(direct)


def test_describe_gives_value_and_rank_and_refuses_features_outside_the_candidates():
    q = _q()
    best = sc.describe(q, [4])                      # only variable 1 (feature 4): value -3 + 0.25, the minimum
    assert best["qubo_value"] == pytest.approx(-2.75) and best["rank"] == 1 and best["n_selected"] == 1
    assert sc.describe(q, [9])["rank"] == 8         # +1.25: the worst of the 8 answers
    assert sc.describe(q, [4, 99]) == {"selected": [4, 99], "n_selected": 2, "qubo_value": None, "rank": None}


def test_summary_counts_and_differences():
    ex = {"selected": [1, 2], "test_ratio": 1.0}
    cl = {"selected": [1, 3], "test_ratio": 1.001}
    rows = [{"seed": 1, "selected": [1, 2], "n_selected": 2, "rank": 1, "test_ratio": 1.0},
            {"seed": 2, "selected": [2, 5], "n_selected": 2, "rank": 40, "test_ratio": 0.999},
            {"seed": 3, "selected": [1, 2], "n_selected": 2, "rank": 1, "test_ratio": 1.0}]
    s = sc.summarise(rows, cl, ex, 65536)
    assert s["qaoa_seeds_finding_the_exact_optimum"] == 2 and s["qaoa_distinct_selections"] == 2 and s["qaoa_rank_worst"] == 40
    assert s["qaoa_seeds_beating_no_change_on_test"] == 1 and s["qaoa_seeds_better_than_classical_on_test"] == 3
    assert s["qaoa_minus_exact_mean"] == pytest.approx(-0.001 / 3) and s["qaoa_minus_exact_ci95"][0] < 0 < s["qaoa_minus_exact_ci95"][1]
    md = sc.to_markdown({"classical": {**cl, "n_selected": 2, "qubo_value": 2.0, "rank": 9, "val_ratio": 1.0},
                         "exact": {**ex, "n_selected": 2, "qubo_value": 1.0, "rank": 1, "val_ratio": 1.0},
                         "qaoa": [{**r, "qubo_value": 1.0, "val_ratio": 1.0} for r in rows], "summary": s})
    assert "EXPLORATORY" in md and "not as a test" in md


def test_it_is_not_on_the_three_arm_path():
    src = open(os.path.join(ROOT, "scripts", "qaoa_selector_comparison.py"), encoding="utf8").read()
    mods = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(k in m for m in mods for k in ("walk_forward", "naive_baseline", "feature_map", "three_arm", "verdicts", "run_hypothesis"))
    for consumer in ("scripts/run_hypothesis_tests.py", "scripts/run_walk_forward.py", "scripts/run_naive_baseline.py",
                     "scripts/run_feature_map_experiment.py", "src/evaluation/three_arm.py", "src/evaluation/verdicts.py"):
        assert "qaoa_selector_comparison" not in open(os.path.join(ROOT, consumer), encoding="utf8").read()
