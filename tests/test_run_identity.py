import numpy as np, pandas as pd
from src.utils.run_identity import config_of_run_id, model_slug, latest_per_config


def test_slug_matches_run_id_prefix():
    assert config_of_run_id("qgan_llm_12_qubit_20261017T095058Z") == model_slug("QGAN-LLM (12-qubit)")
    assert config_of_run_id("qgan_llm_20261016T183043Z") == model_slug("QGAN-LLM")
    assert model_slug("QGAN-LLM") != model_slug("QGAN-LLM (no noise)")


def test_latest_per_config_drops_reevaluations():
    df = pd.DataFrame({"run_id": ["qgan_llm_20261012T234127Z", "qgan_llm_20261016T183043Z",
                                   "qgan_llm_no_noise_20261014T200129Z"], "x": [1, 2, 3]})
    out = latest_per_config(df)
    assert len(out) == 2 and out.loc[out.config == "qgan_llm", "x"].item() == 2


def test_haar_log_space_is_finite_at_20_qubits():
    dim = 2 ** 20
    c = (np.arange(20) + 0.5) / 20
    lp = np.log(dim - 1) + (dim - 2) * np.log1p(-np.clip(c, 0, 1 - 1e-12))
    p = np.exp(lp - lp.max()); p /= p.sum()
    assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-9
