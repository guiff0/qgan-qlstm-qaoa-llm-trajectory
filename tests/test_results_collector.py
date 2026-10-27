"""Regression test: ResultsCollector.export_chapter4_tables must not crash when
several QGAN-LLM ablation configurations share model_type == "QGAN-LLM"
(previously: ValueError: The truth value of a Series is ambiguous)."""
import pandas as pd

from src.experiments.results_collector import ResultsCollector


def _write(tmp_path):
    rows = [
        ("classical_gan_llm_20261016T182958Z", "Classical GAN-LLM", 0.000613, 0.2356),
        ("qgan_llm_20261016T183043Z", "QGAN-LLM", 0.000613, 0.2356),
        ("qgan_llm_12_qubit_20261021T012741Z", "QGAN-LLM", 0.000613, 0.2356),
        ("qgan_llm_linear_topology_20261021T060144Z", "QGAN-LLM", 0.000613, 0.2356),
        ("qgan_llm_no_noise_20261021T234502Z", "QGAN-LLM", 0.000613, 0.2356),
    ]
    pd.DataFrame(rows, columns=["run_id", "model_type", "rmse", "asr"]).to_csv(
        tmp_path / "all_results.csv", index=False)


def test_export_with_duplicate_model_type(tmp_path):
    _write(tmp_path)
    out = ResultsCollector(results_dir=str(tmp_path)).export_chapter4_tables()
    cmp_ = out["primary_vs_baseline"]
    assert cmp_["primary_config"] == "qgan_llm"
    assert cmp_["qgan_rmse"] == 0.000613
    assert cmp_["rmse_reduction_pct"] == 0.0      # 0.0 is reported, not None
    assert cmp_["degenerate_comparison"] is True
    assert len(out["model_comparison"]) == 5
    assert (tmp_path / "chapter4_numbers.json").is_file()
