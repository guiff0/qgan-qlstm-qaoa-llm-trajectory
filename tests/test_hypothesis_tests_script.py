import json

import numpy as np
import pandas as pd

from scripts.run_hypothesis_tests import (
    test_h1_synthetic_data_fidelity as h1_fn,
    test_h2_adversarial_resilience as h2_fn,
    test_h3_poisoning_resistance as h3_fn,
    test_h4_quantum_advantage as h4_fn,
    test_h5_tomography_correlation as h5_fn,
    compute_cqrs_per_row,
)


def _single_row_df():
    return pd.DataFrame([
        {"model_type": "QGAN-LLM", "rmse": 0.9, "asr": 0.3, "fid": 20,
         "entanglement_entropy": 0.8, "accuracy": 0.6},
        {"model_type": "Classical GAN-LLM", "rmse": 1.1, "asr": 0.45, "fid": 28,
         "entanglement_entropy": np.nan, "accuracy": 0.5},
    ])


def _multi_row_df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(5):
        rows.append({"model_type": "QGAN-LLM", "run_id": f"q{i}",
                      "rmse": 0.9 + 0.05 * rng.standard_normal(), "asr": 0.3 + 0.05 * rng.standard_normal(),
                      "fid": 20 + 2 * rng.standard_normal(), "vix": 18 + 2 * rng.standard_normal(),
                      "entanglement_entropy": 0.8 + 0.1 * rng.standard_normal(),
                      "accuracy": 0.6 + 0.05 * rng.standard_normal(),
                      "poisoning_rmse_degradation_pct": 5 + 2 * rng.standard_normal(),
                      "qsfr": 0.9, "eer": 0.95, "nlcs": 0.85, "ctr": np.nan})
    for i in range(5):
        rows.append({"model_type": "Classical GAN-LLM", "run_id": f"c{i}",
                      "rmse": 1.1 + 0.05 * rng.standard_normal(), "asr": 0.45 + 0.05 * rng.standard_normal(),
                      "fid": 28 + 2 * rng.standard_normal(), "vix": 19 + 2 * rng.standard_normal(),
                      "entanglement_entropy": np.nan, "accuracy": 0.5 + 0.05 * rng.standard_normal(),
                      "poisoning_rmse_degradation_pct": 15 + 2 * rng.standard_normal(),
                      "qsfr": np.nan, "eer": np.nan, "nlcs": np.nan, "ctr": np.nan})
    for i in range(3):
        rows.append({"model_type": "QGAN-LLM (no noise)", "run_id": f"nn{i}",
                      "poisoning_rmse_degradation_pct": 22 + 2 * rng.standard_normal(),
                      "entanglement_entropy": 0.3 + 0.1 * rng.standard_normal(),
                      "asr": 0.5 + 0.05 * rng.standard_normal(),
                      "accuracy": 0.48 + 0.05 * rng.standard_normal(),
                      "qsfr": 0.5, "eer": 0.6, "nlcs": 0.4, "ctr": np.nan})
    return pd.DataFrame(rows)


def test_single_run_data_reports_insufficient_data_not_a_crash():
    df = _single_row_df()
    for fn in (h1_fn, h2_fn, h3_fn, h4_fn, h5_fn):
        result = fn(df)
        assert "error" in result or "insufficient data" in json.dumps(result)


def test_multi_run_data_produces_real_statistics():
    df = _multi_row_df()
    h1 = h1_fn(df)
    assert "error" not in h1
    assert "p_value" in h1

    h3 = h3_fn(df)
    assert "error" not in h3
    assert "p_value" in h3

    h4 = h4_fn(df)
    assert "error" not in h4["rmse"]
    assert "p_value" in h4["rmse"]
    assert "caveat" in h4  # same-row-leakage caveat must always be present


def test_qar_computes_real_value_with_asr_clean_columns():
    from scripts.run_hypothesis_tests import test_qar as qar_fn
    df = pd.DataFrame([
        {"model_type": "QGAN-LLM", "asr": 20.0, "asr_clean": 3.0},
        {"model_type": "Classical GAN-LLM", "asr": 45.0, "asr_clean": 8.0},
    ])
    result = qar_fn(df)
    assert "error" not in result
    assert np.isclose(result["qar"], 5.0)  # (45-20)/(8-3)


def test_qar_reports_missing_columns_gracefully():
    from scripts.run_hypothesis_tests import test_qar as qar_fn
    df = pd.DataFrame([{"model_type": "QGAN-LLM", "asr": 20.0}])
    result = qar_fn(df)
    assert "error" in result


def test_h4_does_not_crash_without_rmse_column():
    from scripts.run_hypothesis_tests import test_h4_quantum_advantage as h4_fn
    df = pd.DataFrame([
        {"model_type": "QGAN-LLM", "asr": 20.0},
        {"model_type": "Classical GAN-LLM", "asr": 45.0},
    ])
    result = h4_fn(df)
    assert result["rmse"] == {"error": "no 'rmse' column"}


def test_m2_mediation_runs_without_crashing_on_multi_run_data():
    from scripts.run_hypothesis_tests import test_m2_dv2_mediation as m2_fn
    df = _multi_row_df()
    result = m2_fn(df)
    assert "error" not in result
    assert "indirect_effect" in result
    assert "caveat" in result
    df = _multi_row_df()
    rows = compute_cqrs_per_row(df)
    assert len(rows) == 8  # QGAN-LLM (5) + QGAN-LLM (no noise) (3) have qsfr/eer/nlcs; Classical GAN-LLM doesn't
    for v in rows.values():
        assert np.isfinite(v["CQRS"])
        assert set(v["missing_metrics"]) == {"ctr", "qar"}
