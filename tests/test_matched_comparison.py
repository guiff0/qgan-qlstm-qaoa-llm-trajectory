"""Equal input width + capacity-matched classical control for the generator-path comparison."""
import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils.capacity_match import lstm_generator_params, match_hidden


def test_param_formula_reproduces_the_diagnosed_classical_count():
    assert lstm_generator_params(32, 128, 32) == 87072          # figure in results/no_advantage_diagnosis.json


def test_matched_hidden_is_within_five_percent_of_the_qgan():
    h = match_hidden(560, latent=12, out=32)          # 12-qubit, 4-layer circuit: 560 parameters
    assert h == 5 and abs(lstm_generator_params(12, h, 32) - 560) / 560 < 0.05


def test_config_matches_input_width_and_capacity():
    cfg = yaml.safe_load(open("config/default_config.yaml"))
    assert cfg["classical_gan_llm"]["generator_input_dims"] == cfg["qgan_llm"]["n_qubits"]
    (m,) = [a for a in cfg["classical_ablations"] if a["name"] == "Classical GAN-LLM (matched)"]
    assert m["latent_dim"] == m["generator_input_dims"] == cfg["qgan_llm"]["n_qubits"]
    n = lstm_generator_params(m["latent_dim"], m["generator_hidden"], cfg["classical_gan_llm"].get("output_dim", 32))
    assert abs(n - 560) / 560 < 0.05


def test_input_dims_changes_generator_mode_fingerprint_but_not_head_only():
    pytest.importorskip("torch")
    from src.utils.gan_training import config_fingerprint
    base = {"forecast_path": "generator", "head_fit_rows": 4000, "latent_dim": 32}
    assert config_fingerprint({**base, "generator_input_dims": 20}, 1) != config_fingerprint(base, 1)
    ho = {**base, "forecast_path": "head_only"}
    assert config_fingerprint({**ho, "generator_input_dims": 20}, 1) == config_fingerprint(ho, 1)


def _rows(slug, model, rmses):
    return [{"model_type": model, "run_id": f"{slug}_20261101T0000{i:02d}Z", "rmse": r, "asr": 10 + 1e3 * r,
             "forecast_path": "generator", "persistence_rmse": 1.0} for i, r in enumerate(rmses)]


def test_h4_matched_uses_only_the_matched_rows_and_primary_h4_ignores_them():
    from scripts.run_hypothesis_tests import test_h4_matched_capacity, test_h4_quantum_advantage
    df = pd.DataFrame(_rows("qgan_llm", "QGAN-LLM", [1.0, 1.1, 0.9, 1.05, 0.95])
                      + _rows("classical_gan_llm", "Classical GAN-LLM", [1.2, 1.3, 1.1, 1.25, 1.15])
                      + _rows("classical_gan_llm_matched", "Classical GAN-LLM", [1.0, 1.02, 0.98, 1.01, 0.99]))
    big, small = test_h4_quantum_advantage(df), test_h4_matched_capacity(df)
    assert big["classical_comparator"] == "classical_gan_llm" and big["n_runs"]["Classical GAN-LLM"] == 5
    assert small["classical_comparator"] == "classical_gan_llm_matched" and small["n_runs"]["Classical GAN-LLM"] == 5
    assert big["rmse"] != small["rmse"]


def test_h4_matched_reports_missing_rows_instead_of_guessing():
    from scripts.run_hypothesis_tests import test_h4_matched_capacity
    df = pd.DataFrame(_rows("qgan_llm", "QGAN-LLM", [1.0, 1.1]) + _rows("classical_gan_llm", "Classical GAN-LLM", [1.2, 1.3]))
    assert test_h4_matched_capacity(df)["status"] == "no_matched_rows"


def test_active_dims_zeroes_trailing_features_and_blocks_their_gradient():
    torch = pytest.importorskip("torch")
    from src.utils.generator_path import make_generator_path_head
    seen = {}

    class G(torch.nn.Module):
        def forward(self, z):
            seen["z"] = z.detach().clone()
            return z[:, :4]
    head = make_generator_path_head(G(), 8, active_dims=3)
    head.eval()
    X = torch.randn(5, 8, requires_grad=True)
    head(X).sum().backward()
    assert torch.all(seen["z"][:, 3:] == 0) and torch.any(seen["z"][:, :3] != 0)
