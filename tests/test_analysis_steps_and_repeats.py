"""
Tests for: (1) per-(config, seed) counting of runs, (2) single-case fallback and structured
"insufficient data" results in scripts/run_hypothesis_tests.py, (3) run_pipeline.py's analysis
steps (never skipped, never blocked, runnable alone) and extra-seed steps.
"""
from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_pipeline as rp  # noqa: E402
from scripts import run_hypothesis_tests as rht  # noqa: E402
from src.evaluation.statistical_tests import crawford_howell  # noqa: E402
from src.utils.run_identity import latest_per_config, latest_per_config_seed  # noqa: E402


# ---------------------------------------------------------------- run counting
def _runs_df():
    return pd.DataFrame([
        {"run_id": "classical_gan_llm_20261009T041430Z", "seed": 42, "fid": 100.0},
        {"run_id": "classical_gan_llm_20261025T081104Z", "seed": 42, "fid": 101.0},  # re-eval, same seed
        {"run_id": "classical_gan_llm_20261026T081104Z", "seed": 43, "fid": 99.0},   # independent repeat
    ])


def test_latest_per_config_seed_keeps_each_seed_but_collapses_reevaluations():
    out = latest_per_config_seed(_runs_df())
    assert sorted(out["seed"]) == [42, 43]
    assert out.loc[out["seed"] == 42, "fid"].item() == 101.0      # newest of the two seed-42 rows


def test_latest_per_config_still_collapses_to_one_row():
    assert len(latest_per_config(_runs_df())) == 1


def test_latest_per_config_seed_handles_missing_seed_column():
    df = _runs_df().drop(columns="seed")
    assert len(latest_per_config_seed(df)) == 1


# ---------------------------------------------------------------- statistics
def test_crawford_howell_flags_an_outlying_single_case():
    r = crawford_howell(10.0, np.array([1.0, 1.1, 0.9, 1.05]))
    assert r["p_value"] < 0.001 and r["df"] == 3


def test_compare_two_levels_of_evidence():
    many = np.array([5.0, 5.2, 4.9, 5.1])
    assert rht._compare_two(many, many + 1, "a", "b")["status"] == "ok"
    single = rht._compare_two(np.array([9.0]), many, "a", "b", how_to_fix="run more")
    assert single["status"] == "exploratory_single_case" and "p_value" in single
    none = rht._compare_two(np.array([9.0]), np.array([8.0]), "a", "b", how_to_fix="run more")
    assert none["status"] == "insufficient_data" and "insufficient data" in none["message"]
    assert none["how_to_fix"] == "run more"


def test_h4_reports_degenerate_by_design_not_error():
    df = pd.DataFrame([
        {"model_type": "QGAN-LLM", "rmse": 0.0006, "asr": 0.23},
        {"model_type": "QGAN-LLM", "rmse": 0.0006, "asr": 0.23},
        {"model_type": "Classical GAN-LLM", "rmse": 0.0006, "asr": 0.23},
    ])
    out = rht.test_h4_quantum_advantage(df)
    assert out["rmse"]["status"] == "degenerate_by_design" and "why" in out["rmse"]


def test_sanitize_makes_valid_json():
    txt = json.dumps(rht._sanitize({"qar": float("nan"), "x": np.float64(1.5), "n": np.int64(3)}))
    assert "NaN" not in txt and json.loads(txt)["qar"] is None


# ---------------------------------------------------------------- pipeline
def _setup(monkeypatch, tmp_path, models=("Good Model",)):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rp, "STATE_PATH", os.path.join(str(tmp_path), "data", "pipeline_state.json"))
    monkeypatch.setattr(rp, "resolve_model_names", lambda cfg_path, only: (list(models), None))
    monkeypatch.setattr(rp, "load_config", lambda cfg_path: {
        "seed": 42,
        "data": {"dukascopy_file": str(tmp_path / "d.csv"), "forexsb_file": str(tmp_path / "f.csv"),
                 "fred_file": str(tmp_path / "fr.csv"), "processed_dir": str(tmp_path)},
        "logging": {"results_dir": str(tmp_path)},
    })
    for f in ("d.csv", "f.csv", "fr.csv"):
        (tmp_path / f).write_text("x")
    for prefix in ("X", "y"):
        for split in ("train", "val", "test"):
            (tmp_path / f"{prefix}_{split}.npy").write_bytes(b"x")


class Recorder:
    def __init__(self, tmp_path, fail=()):
        self.calls, self.tmp, self.fail = [], tmp_path, fail

    def __call__(self, cmd, stdout=None, stderr=None, env=None):
        joined = " ".join(cmd)
        self.calls.append(joined)
        if any(f in joined for f in self.fail):
            return types.SimpleNamespace(returncode=1)
        if "Good Model" in joined:
            (self.tmp / "all_results.csv").write_text("run_id,seed\ngood_model_20261001T000000Z,42\n")
        if "verify_chapter4" in joined:
            (self.tmp / "chapter4_numbers.json").write_text("{}")
        if "run_hypothesis_tests" in joined:
            (self.tmp / "hypothesis_test_results.json").write_text("{}")
        return types.SimpleNamespace(returncode=0)


def test_stats_only_runs_just_hypothesis_tests_and_nothing_else(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(rp, "resolve_model_names",
                        lambda *a: (_ for _ in ()).throw(AssertionError("must not list models")))
    rec = Recorder(tmp_path)
    monkeypatch.setattr(rp.subprocess, "run", rec)
    assert rp.run_pipeline(["--stats-only"]) == 0
    assert len(rec.calls) == 1 and "scripts.run_hypothesis_tests" in rec.calls[0]


def test_hypothesis_tests_rerun_even_when_completed_and_yes_given(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    rec = Recorder(tmp_path)
    monkeypatch.setattr(rp.subprocess, "run", rec)
    rp.run_pipeline(["--yes"])
    first = sum("run_hypothesis_tests" in c for c in rec.calls)
    rp.run_pipeline(["--yes"])          # everything is now "completed"; the stats must still re-run
    second = sum("run_hypothesis_tests" in c for c in rec.calls)
    assert first == 1 and second == 2
    train_calls = [c for c in rec.calls if "run_all" in c]
    assert len(train_calls) == 1        # the training step WAS skipped the second time


def test_hypothesis_tests_rerun_after_a_previous_failure(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(rp.subprocess, "run", Recorder(tmp_path, fail=("run_hypothesis_tests",)))
    rp.run_pipeline(["--yes"])
    assert rp.load_state()["steps"]["run_hypothesis_tests"]["status"] == "failed"
    rec = Recorder(tmp_path)
    monkeypatch.setattr(rp.subprocess, "run", rec)
    rp.run_pipeline(["--yes"])
    assert rp.load_state()["steps"]["run_hypothesis_tests"]["status"] == "completed"
    assert any("run_hypothesis_tests" in c for c in rec.calls)


def test_failed_train_step_does_not_block_analysis(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, models=("Good Model", "Bad Model"))
    rec = Recorder(tmp_path, fail=("Bad Model",))
    monkeypatch.setattr(rp.subprocess, "run", rec)
    assert rp.run_pipeline(["--yes"]) == 1
    assert any("run_hypothesis_tests" in c for c in rec.calls)


def test_failing_verify_chapter4_does_not_block_hypothesis_tests(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    rec = Recorder(tmp_path, fail=("verify_chapter4",))
    monkeypatch.setattr(rp.subprocess, "run", rec)
    assert rp.run_pipeline(["--yes"]) == 1
    assert any("run_hypothesis_tests" in c for c in rec.calls)


def test_required_upstream_failure_still_refreshes_the_stats(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    rec = Recorder(tmp_path, fail=("dukascopy",))
    monkeypatch.setattr(rp.subprocess, "run", rec)
    assert rp.run_pipeline(["--yes"]) == 1
    assert any("run_hypothesis_tests" in c for c in rec.calls)
    assert not any("prepare_data" in c for c in rec.calls)


def test_extra_seeds_adds_one_step_per_model_and_seed(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, models=("Good Model", "Other"))
    rec = Recorder(tmp_path)
    monkeypatch.setattr(rp.subprocess, "run", rec)
    rp.run_pipeline(["--yes", "--extra-seeds", "2", "--seed-models", "Good Model"])
    seed_calls = [c for c in rec.calls if "--seed" in c]
    assert len(seed_calls) == 2 and "--seed 43" in seed_calls[0] and "--seed 44" in seed_calls[1]


def test_unknown_step_id_is_rejected(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    import pytest
    with pytest.raises(SystemExit):
        rp.run_pipeline(["--steps", "nope"])
