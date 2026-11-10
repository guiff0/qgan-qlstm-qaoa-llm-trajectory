"""H1/H3 must compare ONE condition per side (not pool QGAN-LLM ablations), and the QLSTM's persistence
RMSE must be computed on the rows its model RMSE was scored on. Pure numpy/pandas; no torch/PennyLane."""
import ast
import pathlib

import numpy as np
import pandas as pd

import scripts.run_hypothesis_tests as hyp
from src.utils.generator_path import forecast_eval_indices, persistence_rmse_on

TS = "_20261031T120000Z"


def _rows(slug, model_type, n, fid=None, deg=None, noise=0.01, seed0=43, rng=None):
    out = []
    for k in range(n):
        r = {"run_id": f"{slug}_20261031T12{k:02d}00Z", "model_type": model_type, "seed": seed0 + k,
             "noise_strength": noise}
        if fid is not None:
            r["fid"] = fid + rng.normal(0, 0.5)
        if deg is not None:
            r["poisoning_rmse_degradation_pct"] = deg + rng.normal(0, 0.5)
        out.append(r)
    return out


def test_h1_uses_primary_configs_only():
    rng = np.random.default_rng(0)
    rows = (_rows("qgan_llm", "QGAN-LLM", 4, fid=20, rng=rng)
            + _rows("classical_gan_llm", "Classical GAN-LLM", 4, fid=28, rng=rng)
            # ablations with wildly different FID must not leak into the QGAN group
            + _rows("qgan_llm_12_qubit", "QGAN-LLM", 3, fid=200, rng=rng)
            + _rows("qgan_llm_ent_l0_none_d4", "QGAN-LLM", 3, fid=-100, rng=rng))
    r = hyp.test_h1_synthetic_data_fidelity(pd.DataFrame(rows))
    assert r["n_per_group_used"] == {"QGAN-LLM": 4, "Classical GAN-LLM": 4}
    assert r["pooled_qgan_configs"] is False
    assert r["p_value"] < 0.01 and r["configs_used"]["QGAN-LLM"] == "qgan_llm"


def test_h1_flags_pooling_when_no_primary_config_exists():
    rng = np.random.default_rng(1)
    rows = (_rows("qgan_llm_12_qubit", "QGAN-LLM", 2, fid=20, rng=rng)
            + _rows("qgan_llm_linear_topology", "QGAN-LLM", 2, fid=22, rng=rng)
            + _rows("classical_gan_llm", "Classical GAN-LLM", 3, fid=28, rng=rng))
    r = hyp.test_h1_synthetic_data_fidelity(pd.DataFrame(rows))
    assert r["pooled_qgan_configs"] is True          # verdict layer then downgrades to exploratory
    assert "no primary-config rows" in r["configs_used"]["QGAN-LLM"]


def test_h3_compares_primary_against_no_noise_only():
    rng = np.random.default_rng(2)
    rows = (_rows("qgan_llm_no_noise", "QGAN-LLM", 4, deg=22, noise=0.0, rng=rng)
            + _rows("qgan_llm", "QGAN-LLM", 4, deg=5, rng=rng)
            + _rows("qgan_llm_12_qubit", "QGAN-LLM", 3, deg=500, rng=rng)
            + _rows("qgan_llm_linear_topology", "QGAN-LLM", 3, deg=-500, rng=rng)
            + _rows("qgan_llm_ent_l6_full_d4", "QGAN-LLM", 3, deg=900, rng=rng))
    r = hyp.test_h3_poisoning_resistance(pd.DataFrame(rows))
    assert r["design"] == "matched_config_noise_only"
    assert r["n_per_group_used"] == {"no_noise": 4, "noisy": 4}
    assert "confound_warning" not in r                # nothing left to warn about
    assert r["t_statistic"] > 0 and r["p_value"] < 0.01


def test_h3_is_untestable_not_pooled_when_the_matched_config_is_missing():
    rng = np.random.default_rng(3)
    rows = (_rows("qgan_llm_no_noise", "QGAN-LLM", 3, deg=22, noise=0.0, rng=rng)
            + _rows("qgan_llm_12_qubit", "QGAN-LLM", 3, deg=5, rng=rng))      # no primary 'qgan_llm' rows
    r = hyp.test_h3_poisoning_resistance(pd.DataFrame(rows))
    # reviewer round 5: no pooled / cross-design fallback; the verdict is UNTESTABLE
    assert r["status"] == "insufficient_data" and r["design"] == "matched_pair_missing"


def test_qlstm_persistence_is_scored_on_the_model_rows():
    rng = np.random.default_rng(4)
    last = np.cumsum(rng.normal(0, 1e-4, 5000)) + 1.1
    y_next = last + rng.normal(0, 1e-4, 5000) * (1 + 5 * (np.arange(5000) > 4000))   # heteroscedastic tail
    sel = forecast_eval_indices(5000, 500, seed=43)
    sub = persistence_rmse_on(last, y_next, sel)
    full = persistence_rmse_on(last, y_next, None)
    assert sub == np.sqrt(np.mean((y_next[sel] - last[sel]) ** 2))
    assert not np.isclose(sub, full)                  # the old full-set fallback would have differed


def test_qlstm_evaluate_reports_its_own_persistence_on_the_same_selection():
    src = pathlib.Path("src/baselines/qlstm_forecaster.py").read_text()
    tree = ast.parse(src)
    ev = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    code = ast.get_source_segment(src, ev)
    assert '"persistence_rmse": persistence_rmse_on(_te.last, _te.y_next, sel)' in code
    assert "sel = None" in code                       # `sel` exists even when no subsampling happens
