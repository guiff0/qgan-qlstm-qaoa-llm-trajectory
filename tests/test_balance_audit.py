import copy
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.check_balance import run  # noqa: E402
from src.evaluation.balance_audit import audit, compute_budget, format_report, resolve_arms  # noqa: E402


@pytest.fixture
def cfg():
    with open(os.path.join(ROOT, "config", "default_config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def failed(rep):
    return {c["check"] for c in rep["checks"] if not c["ok"]}


def test_real_config_is_balanced(cfg):
    rep = audit(cfg)
    assert rep["pass"], format_report(rep)
    row = {r["arm"]: r for r in rep["param_table"]}
    assert row["QGAN-LLM"]["generator_params"] == 560 and row["Classical GAN-LLM (matched)"]["generator_params"] == 572
    assert row["Classical GAN-LLM"]["ratio_to_matched_classical"] > 90          # the bracket is disclosed, not hidden


def test_arms_are_resolved_like_run_all(cfg):
    arms = resolve_arms(cfg)
    assert arms["QGAN-LLM (no noise)"]["noise_strength"] == 0.0 and arms["QGAN-LLM (no noise)"]["epochs"] == 30
    assert arms["Classical GAN-LLM (matched)"]["generator_hidden"] == 5


@pytest.mark.parametrize("mutate,expect", [
    (lambda c: c["classical_gan_llm"].update(epochs=60), "shared_keys_identical"),
    (lambda c: c["qgan_llm"].update(learning_rate=1e-3), "shared_keys_identical"),
    (lambda c: c["qgan_llm"].update(alpha=10.0), "loss_weight_equal"),            # the "100-fold loss weight" case
    (lambda c: c["classical_gan_llm"].update(mse_weight=0.001), "loss_weight_equal"),
    (lambda c: (c["qgan_llm"].update(max_batches_per_epoch=5), c["classical_gan_llm"].update(max_batches_per_epoch=5),
                c["qgan_llm"].update(batch_size=8)), "rows_seen_equal"),
    (lambda c: c["classical_ablations"][0].update(generator_hidden=64), "params_matched"),
    (lambda c: c["classical_ablations"][0].update(generator_input_dims=32), "input_information_equal"),
    (lambda c: c["qgan_llm"].update(hpo_trials=50), "no_per_arm_tuning"),
])
def test_each_imbalance_is_caught(cfg, mutate, expect):
    bad = copy.deepcopy(cfg)
    mutate(bad)
    rep = audit(bad)
    assert not rep["pass"] and expect in failed(rep)


def test_missing_matched_arm_fails(cfg):
    bad = copy.deepcopy(cfg)
    bad["classical_ablations"] = []
    assert not audit(bad)["pass"]


def test_compute_budget_from_runs():
    runs = [{"status": "ok", "slug": "qgan_llm", "minutes": 2800}, {"status": "ok", "slug": "qgan_llm", "minutes": 2900},
            {"status": "ok", "slug": "classical_gan_llm_matched", "minutes": 20},
            {"status": "failed", "slug": "classical_gan_llm_matched", "minutes": 0.6},
            {"status": "ok", "slug": "classical_gan_llm_matched", "minutes": 22, "adopted": True}]
    out = compute_budget(runs)
    assert out["median_minutes_per_condition"]["qgan_llm"] == 2850 and out["quantum_over_matched_classical"] == 142.5


def test_run_writes_report_into_the_experiment_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rep = run(os.path.join(ROOT, "config", "default_config.yaml"), "EXP-20261107-WINBOX-aaaa")
    assert rep["pass"] and os.path.isfile(tmp_path / "results" / "experiments" / "EXP-20261107-WINBOX-aaaa" / "balance_audit.json")
