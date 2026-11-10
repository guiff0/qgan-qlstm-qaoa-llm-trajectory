import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import three_arm
from src.quantum.feature_map import ClassicalRandomFeatureMap


def _rows(rmse, pers=1.0, n=10, ridge=1.0):
    rows, dm = [], []
    for s in range(n):
        for arm, r in rmse.items():
            rows.append({"arm": arm, "seed": s, "rmse": r, "persistence_rmse": pers, "skill": 1 - (r / pers) ** 2,
                         "ridge": ridge, "n_features": 40 if arm != "linear" else 32, "n_test": 1000})
    return rows


def _dm(stat, p, n=10):
    return [{"seed": s, "vs": c, "statistic": stat, "p_value": p} for s in range(n) for c in ("linear", "classical_rf")]


def test_classical_control_never_exceeds_the_circuit_parameter_budget():
    for n, l in ((4, 3), (8, 3), (12, 3), (12, 2), (16, 3)):
        c = ClassicalRandomFeatureMap(n, n, n * l * 3, seed=0)
        assert c.n_random_params <= n * l * 3 and c.n_random_params >= 0.95 * n * l * 3
        assert c.transform(np.random.randn(4, n)).shape == (4, n)


def test_advantage_needs_both_controls_and_skill_above_zero():
    rows = _rows({"linear": 0.99, "classical_rf": 0.99, "quantum": 0.95})
    assert three_arm.verdict(rows, _dm(3.0, 0.001))["verdict"] == "ADVANTAGE_SUPPORTED"
    # quantum beats the controls but nothing beats the naive baseline -> flagged, and no ADVANTAGE
    bad = _rows({"linear": 1.05, "classical_rf": 1.05, "quantum": 1.02})
    v = three_arm.verdict(bad, _dm(3.0, 0.001))
    assert v["verdict"] != "ADVANTAGE_SUPPORTED" and any("persistence" in n for n in v["notes"])


def test_equivalence_contradiction_and_underpowered():
    same = _rows({"linear": 0.99, "classical_rf": 0.99, "quantum": 0.99}, n=10)
    noisy = _rows({"linear": 0.99, "classical_rf": 0.99, "quantum": 0.99}, n=10)
    for i, r in enumerate(noisy):   # tiny paired differences around 0 -> equivalent within 2 % of persistence
        if r["arm"] == "quantum":
            r["rmse"] += 0.001 * (-1) ** i
    assert three_arm.verdict(noisy, _dm(0.0, 0.9))["verdict"] == "EQUIVALENT_TO_CONTROLS"
    worse = _rows({"linear": 0.95, "classical_rf": 0.95, "quantum": 0.99})
    assert three_arm.verdict(worse, _dm(-3.0, 0.001))["verdict"] == "CONTRADICTED"
    assert three_arm.verdict(_rows({"linear": .99, "classical_rf": .99, "quantum": .95}, n=4), _dm(3.0, 0.001, n=4))["verdict"] == "UNDERPOWERED"


def test_audit_flags_unbalanced_seeds_budget_and_grid_edge():
    rows = _rows({"linear": 0.99, "classical_rf": 0.99, "quantum": 0.98}, ridge=1.0)
    grid = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0)
    ok = three_arm.audit(rows, {"n_qubits": 12, "n_layers": 3}, grid)
    assert ok["all_passed"] and {x["dimension"] for x in ok["not_applicable"]} == {"synthetic-sample count", "loss weights"}
    edge = three_arm.audit(_rows({"linear": .99, "classical_rf": .99, "quantum": .98}, ridge=100.0), {"n_qubits": 8, "n_layers": 3}, grid)
    assert not edge["all_passed"]
    short = [r for r in rows if not (r["arm"] == "quantum" and r["seed"] == 0)]
    assert not three_arm.audit(short, {"n_qubits": 8, "n_layers": 3}, grid)["all_passed"]


def test_matrix_cell_runs_end_to_end_on_synthetic_data():
    from scripts.run_three_arm_matrix import cells, run_cell
    rng = np.random.default_rng(0)
    n = 900
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, n))
    X = rng.normal(size=(n, 6))
    data = (X[:500], close[:500], X[500:700], close[500:700], X[700:], close[700:])
    r = run_cell(data, seeds=[0, 1], n_qubits=4, n_layers=2)
    assert set(r["skill"]) == set(three_arm.ARMS) and r["verdict"] == "UNDERPOWERED"      # 2 seeds < 10: descriptive only
    b = run_cell(data, seeds=[0, 1], n_qubits=4, n_layers=2, classical_width_mult=2, classical_grid=12)
    assert b["verdict"] == "UNDERPOWERED"
    names = [c[0] for c in cells(8)]
    assert any(n.startswith("BRACKET") for n in names) and any("train_frac" in n for n in names) and any("ridge_grid" in n for n in names)


def test_naive_baseline_block_in_hypothesis_script():
    from scripts.run_hypothesis_tests import test_naive_baseline
    df = pd.DataFrame({"run_id": [f"qgan_llm_2026110{i % 3}T000000Z" for i in range(6)] + [f"classical_gan_llm_2026110{i % 3}T000000Z" for i in range(6)],
                       "rmse_over_persistence": [0.9, 0.92, 0.91, 0.9, 0.93, 0.92] + [1.2, 1.25, 1.22, 1.3, 1.21, 1.24]})
    out = test_naive_baseline(df)["rmse_over_persistence"]["configs"]
    assert out["qgan_llm"]["beats_naive"] is True and out["qgan_llm"]["ratio_mean"] < 1
    assert out["classical_gan_llm"]["beats_naive"] is False and out["classical_gan_llm"]["ratio_mean"] > 1
