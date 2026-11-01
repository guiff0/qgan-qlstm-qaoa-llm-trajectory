"""Generator on the forecast path (forecast_path: generator).

The numpy / config / fingerprint / H4-row-selection tests need no torch. The torch tests build tiny
models (4 qubits, 8 features) exactly like tests/test_smoke.py and are skipped if torch or pennylane is
missing. They were written against the code but could not be executed in the sandbox this change was made in
(no torch there), so run this file once: `pytest tests/test_generator_path.py -v`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils.generator_path import (
    forecast_eval_indices, persistence_rmse_on, ridge_fit, use_generator_path,
)
from src.utils.gan_training import config_fingerprint

N_FEATURES = 8


# --------------------------------------------------------------------------- numpy
def test_ridge_fit_recovers_weights_and_survives_collinear_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    w0 = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ w0 + 0.7 + rng.normal(0, 0.01, 2000)
    w, b, mse = ridge_fit(X, y)
    assert np.abs(w - w0).max() < 0.01 and abs(b - 0.7) < 0.01 and mse < 1e-3
    F = np.hstack([X, X @ rng.normal(size=(5, 4))])            # 9 columns, rank 5: singular normal equations
    w2, b2, mse2 = ridge_fit(F, y)
    assert np.isfinite(w2).all() and np.abs(w2).max() < 20 and mse2 < 2e-3


def test_eval_subsample_is_deterministic_and_shared_between_models():
    a = forecast_eval_indices(10_000, 2_000, 43)
    assert (a == forecast_eval_indices(10_000, 2_000, 43)).all() and len(a) == 2_000 and (np.diff(a) > 0).all()
    assert forecast_eval_indices(100, None, 1) is None and forecast_eval_indices(100, 500, 1) is None
    assert not (a == forecast_eval_indices(10_000, 2_000, 44)).all()


def test_persistence_is_scored_on_the_same_rows():
    last, nxt = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0])
    assert persistence_rmse_on(last, nxt) == pytest.approx(np.sqrt((1 + 0 + 4) / 3))
    assert persistence_rmse_on(last, nxt, np.array([0, 2])) == pytest.approx(np.sqrt((1 + 4) / 2))


def test_forecast_path_validation():
    assert use_generator_path({"forecast_path": "generator"}) and not use_generator_path({})
    with pytest.raises(ValueError):
        use_generator_path({"forecast_path": "nope"})


# ------------------------------------------------------------------- config parity
def test_both_gan_sections_use_identical_forecast_path_settings():
    cfg = yaml.safe_load(open("config/default_config.yaml"))
    keys = ("forecast_path", "head_fit_rows", "forecast_eval_samples", "attack_samples", "forecast_chunk")
    for k in keys:
        assert cfg["qgan_llm"][k] == cfg["classical_gan_llm"][k], (
            f"{k} differs between qgan_llm and classical_gan_llm: a quantum-vs-classical gap would then be a "
            f"difference in setup, not in the generator")
    assert cfg["qgan_llm"]["forecast_path"] == "generator"


# ------------------------------------------------------------------- fingerprint
def test_head_only_fingerprint_is_unchanged_by_the_new_keys():
    base = {"n_qubits": 4, "epochs": 2}
    legacy = config_fingerprint(base, 42, "qgan_llm")
    with_keys = {**base, "forecast_path": "head_only", "head_fit_rows": 4000, "forecast_eval_samples": 2000,
                 "attack_samples": 50, "forecast_chunk": 256}
    assert config_fingerprint(with_keys, 42, "qgan_llm") == legacy          # old 12 h checkpoints still resume


def test_generator_mode_gets_its_own_fingerprint_and_eval_knobs_do_not_change_it():
    base = {"n_qubits": 4, "epochs": 2}
    gen = {**base, "forecast_path": "generator", "head_fit_rows": 4000}
    assert config_fingerprint(gen, 42, "qgan_llm") != config_fingerprint(base, 42, "qgan_llm")
    assert config_fingerprint({**gen, "head_fit_rows": 1000}, 42, "qgan_llm") != config_fingerprint(gen, 42, "qgan_llm")
    same = {**gen, "forecast_eval_samples": 5, "attack_samples": 3, "forecast_chunk": 8}
    assert config_fingerprint(same, 42, "qgan_llm") == config_fingerprint(gen, 42, "qgan_llm")


# ----------------------------------------------------------------------------- H4
def _rows(slug, model_type, rmses, path):
    return [{"run_id": f"{slug}_20261031T0000{i:02d}Z", "model_type": model_type, "seed": 43 + i,
             "rmse": r, "asr": 10.0 + r * 1e3, "forecast_path": path} for i, r in enumerate(rmses)]


def test_h4_compares_only_generator_path_primary_configs():
    import scripts.run_hypothesis_tests as hyp
    df = pd.DataFrame(
        _rows("qgan_llm", "QGAN-LLM", [1.0e-4, 1.1e-4, 0.9e-4], "generator")
        + _rows("classical_gan_llm", "Classical GAN-LLM", [2.0e-4, 2.2e-4, 1.9e-4], "generator")
        # contamination that must be ignored: sweep + ablation rows share model_type == "QGAN-LLM"
        + _rows("qgan_llm_ent_l6_full_d4", "QGAN-LLM", [9e-4] * 6, "generator")
        + _rows("qgan_llm_no_noise", "QGAN-LLM", [8e-4], "generator")
        # legacy head-only rows that must be ignored once generator-path rows exist for both models
        + _rows("qgan_llm", "QGAN-LLM", [5e-4], "head_only"))
    r = hyp.test_h4_quantum_advantage(df)
    assert r["forecast_design"] == "generator_on_forecast_path"
    assert r["n_runs"] == {"QGAN-LLM": 3, "Classical GAN-LLM": 3}
    assert r["rmse"].get("status") != "degenerate_by_design" and "p_value" in r["rmse"]
    # means are over the three primary runs only; the 7 sweep/ablation rows (9e-4, 8e-4) and the head-only row
    # (5e-4) would drag mean_a far above 1e-4 if they leaked in
    assert r["rmse"]["mean_a"] == pytest.approx(1.0e-4, rel=1e-6)
    assert r["rmse"]["mean_b"] == pytest.approx(2.0333333e-4, rel=1e-5) and r["rmse"]["df"] == 4


def test_h4_without_generator_path_rows_stays_labelled_head_only():
    import scripts.run_hypothesis_tests as hyp
    df = pd.DataFrame(_rows("qgan_llm", "QGAN-LLM", [1e-4, 1e-4], "head_only")
                      + _rows("classical_gan_llm", "Classical GAN-LLM", [1e-4, 1e-4], "head_only"))
    r = hyp.test_h4_quantum_advantage(df)
    assert r["forecast_design"] == "head_only_or_mixed" and r["rmse"]["status"] == "degenerate_by_design"


# ----------------------------------------------------------------------------- torch
torch = pytest.importorskip("torch")


def _toy_generator():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(N_FEATURES, N_FEATURES), torch.nn.Tanh())


def test_head_state_dict_excludes_the_generator_and_eval_propagates():
    from src.utils.generator_path import make_generator_path_head
    g = _toy_generator()
    head = make_generator_path_head(g, N_FEATURES)
    keys = set(head.state_dict())
    assert keys == {"linear.weight", "linear.bias", "z_mean", "z_std"}          # no duplicated generator weights
    assert len(list(head.parameters())) == 2
    g.train()
    head.eval()
    assert not g.training                                                       # training-time noise switched off


def test_fit_uses_the_generator_features_and_beats_a_pure_x_fit():
    from src.utils.generator_path import fit_generator_path_head, make_generator_path_head
    g = _toy_generator()
    head = make_generator_path_head(g, N_FEATURES)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, N_FEATURES)).astype(np.float32)
    with torch.no_grad():                                                       # target that IS a function of G(standardised X)
        z = torch.from_numpy((X - X.mean(0)) / X.std(0))
        y = (2.0 * g(z)[:, 0] + 0.5 * g(z)[:, 3]).numpy()
    mse_gen = fit_generator_path_head(head, X, y, n_rows=600, seed=0, chunk=64)
    _, _, mse_x = ridge_fit(X.astype(np.float64), y)
    assert mse_gen < 0.05 * mse_x, f"G(X) features did not help: {mse_gen:.3e} vs X-only {mse_x:.3e}"


def test_chunked_forward_matches_one_shot_and_gradients_reach_the_input():
    from src.utils.generator_path import chunked_forward, make_generator_path_head
    head = make_generator_path_head(_toy_generator(), N_FEATURES)
    head.eval()
    X = torch.randn(50, N_FEATURES)
    with torch.no_grad():
        full = head(X)
    assert torch.allclose(chunked_forward(head, X, chunk=7), full, atol=1e-6)
    Xg = X[:5].clone().requires_grad_(True)                                     # what FGSM/PGD/CW need
    head(Xg).sum().backward()
    assert Xg.grad is not None and torch.any(Xg.grad != 0)


def _data(n=400):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    y = np.cumsum(0.1 * X[:, 0] + 0.05 * rng.normal(size=n)).astype(np.float32)
    s = n // 2
    return X[:s], y[:s], X[s:s + n // 4], y[s:s + n // 4], X[s + n // 4:], y[s + n // 4:]


GEN_CFG = {"forecast_path": "generator", "head_fit_rows": 60, "forecast_eval_samples": 40, "attack_samples": 6,
           "forecast_chunk": 16, "epochs": 2, "batch_size": 16, "discriminator_hidden": 8}
ATTACK = {"fgsm_epsilon": 0.1, "pgd_epsilon": 0.1, "pgd_alpha": 0.01, "pgd_steps": 2, "cw_c": 1.0,
          "cw_steps": 2, "attacks": ["fgsm"]}


def test_classical_gan_generator_path_trains_resumes_and_scores_a_subsample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.baselines.classical_gan_llm import ClassicalGANLLM
    Xtr, ytr, Xva, yva, Xte, yte = _data()
    cfg = {**GEN_CFG, "latent_dim": N_FEATURES, "output_dim": N_FEATURES, "generator_hidden": 8}
    m = ClassicalGANLLM(config=cfg)
    m.train(Xtr, ytr, Xva, yva)
    res = m.evaluate(Xte, yte, attack_cfg=ATTACK, last_input_prices=Xte[:, 0])
    assert res["forecast_path"] == "generator" and res["forecast_eval_n"] == 40 and res["attack_n_samples"] == 6
    assert np.isfinite(res["rmse"]) and np.isfinite(res["persistence_rmse"]) and 0 <= res["asr"] <= 100
    pred = m.predict(Xte[:20])
    m2 = ClassicalGANLLM(config=cfg)                                             # same config + seed -> resumes
    m2.train(Xtr, ytr, Xva, yva)
    assert m2.resumed_from_checkpoint
    assert np.allclose(m2.predict(Xte[:20]), pred, atol=1e-5)                    # head + standardiser restored
    legacy = ClassicalGANLLM(config={**cfg, "forecast_path": "head_only"})
    legacy.train(Xtr, ytr, Xva, yva)
    assert not np.allclose(legacy.predict(Xte[:20]), pred, atol=1e-6)            # the path really changes the forecast


def test_qgan_generator_path_puts_the_circuit_on_the_inference_path(tmp_path, monkeypatch):
    pytest.importorskip("pennylane")
    monkeypatch.chdir(tmp_path)
    from src.baselines.qgan_llm import QGANLLM
    Xtr, ytr, Xva, yva, Xte, yte = _data()
    cfg = {**GEN_CFG, "n_qubits": 4, "n_layers": 1, "n_features": N_FEATURES, "entanglement": "ring",
           "noise_strength": 0.01}
    m = QGANLLM(config=cfg)
    m.train(Xtr, ytr, Xva, yva)
    # the circuit parameters change the forecast: perturb theta, predictions must move
    before = m.predict(Xte[:10]).copy()
    with torch.no_grad():
        m.generator.theta.add_(0.5)
    assert not np.allclose(m.predict(Xte[:10]), before, atol=1e-6)
    res = m.evaluate(Xte, yte, attack_cfg=ATTACK, last_input_prices=Xte[:, 0])
    assert res["forecast_path"] == "generator" and res["entanglement_entropy"] >= 0
