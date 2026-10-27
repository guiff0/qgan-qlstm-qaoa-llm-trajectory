import pandas as pd

from src.experiments.results_collector import ResultsCollector


def test_export_chapter4_tables_handles_many_qgan_variants(tmp_path):
    """Regression: all QGAN-LLM ablations share model_type, which crashed
    export_chapter4_tables with a Series-truth-value error."""
    rows = [
        dict(run_id="classical_gan_llm_20261016T182958Z", model_type="Classical GAN-LLM", rmse=0.000613, asr=0.236),
        dict(run_id="qgan_llm_20261016T183043Z", model_type="QGAN-LLM", rmse=0.000613, asr=0.236),
        dict(run_id="qgan_llm_12_qubit_20261021T012741Z", model_type="QGAN-LLM", rmse=0.000613, asr=0.236),
        dict(run_id="qgan_llm_no_noise_20261025T085629Z", model_type="QGAN-LLM", rmse=0.000613, asr=0.236),
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "all_results.csv", index=False)
    c = ResultsCollector(str(tmp_path))
    t = c.export_chapter4_tables()
    assert t["primary_vs_baseline"]["rmse_reduction_pct"] == 0.0
    assert t["gan_rmse_degenerate"] is True
