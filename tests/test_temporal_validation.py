import json
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import run_temporal_validation as rtv  # noqa: E402
from src.evaluation import temporal_validation as tv  # noqa: E402
from src.utils import prediction_dump as pd_  # noqa: E402

N_TEST, N_ROWS = 20000, 2000


def make_arrs(skill_noise=0.8, seed=0, n=N_ROWS, shared_rows_seed=43, vix=True):
    rng = np.random.default_rng(seed)
    idx = np.sort(np.random.default_rng(shared_rows_seed).choice(N_TEST, n, replace=False))
    y = rng.normal(0, 1e-4, n)
    a = {"row_index": idx, "y_true": y, "y_pred": y + rng.normal(0, 1e-4 * skill_noise, n) * 0 + (y * 0.3) + rng.normal(0, 1e-5, n),
         "y_persist": np.zeros(n), "n_rows_test_shifted": N_TEST}
    if vix:
        a["vix"] = np.where(idx < N_TEST * 0.3, 13.0, np.where(idx < N_TEST * 0.8, 18.0, 30.0))
    return a


def test_overall_and_block_skill_are_below_one_for_a_model_that_beats_persistence():
    a = make_arrs()
    s = tv.overall_skill(a)
    assert 0 < s["ratio"] < 1 and s["n"] == N_ROWS
    blocks = tv.block_skill(a, 10)
    assert len(blocks) == 10 and sum(b["n"] for b in blocks) == N_ROWS and all(b["ratio"] < 1 for b in blocks)
    reg = {r["regime"]: r for r in tv.regime_skill(a)}
    assert set(reg) == {"low", "normal", "high"} and all(v["ratio"] < 1 for v in reg.values())


def test_model_equal_to_persistence_has_ratio_one():
    a = make_arrs()
    a["y_pred"] = a["y_persist"].copy()
    assert tv.overall_skill(a)["ratio"] == pytest.approx(1.0)


def test_bootstrap_ci_contains_the_point_estimate_and_flags_beating_persistence():
    a = make_arrs()
    b = tv.block_bootstrap_ratio(a, n_boot=300)
    assert b["ci_low"] <= b["ratio"] <= b["ci_high"] and b["beats_persistence"] is True
    assert tv.block_bootstrap_ratio({**a, "row_index": a["row_index"][:20], "y_true": a["y_true"][:20],
                                     "y_pred": a["y_pred"][:20], "y_persist": a["y_persist"][:20]})["status"] == "insufficient_data"


def test_dm_requires_identical_rows_and_orients_the_sign():
    good, worse = make_arrs(seed=1), make_arrs(seed=1)
    worse["y_pred"] = worse["y_true"] * 0.05 + np.random.default_rng(9).normal(0, 3e-5, N_ROWS)   # larger errors
    r = tv.dm_same_rows(good, worse)
    assert r["status"] == "ok" and r["a_better"] is True and r["p_value"] < 0.05
    other_rows = make_arrs(seed=1, shared_rows_seed=99)
    assert tv.dm_same_rows(good, other_rows)["status"] == "rows_differ"


def test_advantage_rule_needs_8_of_10_seeds_and_skill_below_one():
    win = {"status": "ok", "a_better": True, "p_value": 0.01, "statistic": -2.5}
    lose = {"status": "ok", "a_better": False, "p_value": 0.01, "statistic": 2.5}
    seeds = list(range(43, 53))
    r = tv.advantage_rule({s: (win if i < 8 else lose) for i, s in enumerate(seeds)}, {s: 0.9 for s in seeds})
    assert r["rule_met"] and r["seeds_where_quantum_wins_significantly"] == 8 and r["needed"] == 8
    r = tv.advantage_rule({s: (win if i < 7 else lose) for i, s in enumerate(seeds)}, {s: 0.9 for s in seeds})
    assert not r["rule_met"]
    r = tv.advantage_rule({s: win for s in seeds}, {s: 1.2 for s in seeds})       # wins, but loses to persistence
    assert not r["rule_met"] and not r["persistence_skill_positive"]


class FakeModel:
    _gen_path, seed = True, 43
    config = {"forecast_target": "delta", "forecast_eval_samples": 50}

    def predict(self, X):
        return X[:, 0] * 0.5


def test_dump_writes_matching_predictions_and_the_loader_reads_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EXPERIMENT_ID", "EXP-20261107-WINBOX-aaaa")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = np.cumsum(rng.normal(size=400))
    splits = {"X_test": X, "y_test": y}
    from src.evaluation.one_step_ahead import make_target
    te = make_target(X, y, "delta")
    from src.utils.generator_path import forecast_eval_indices
    sel = forecast_eval_indices(len(te.X), 50, 43)
    logged = float(np.sqrt(np.mean((np.asarray(te.target)[sel] - te.X[sel][:, 0] * 0.5) ** 2)))
    path = pd_.dump_eval_predictions(FakeModel(), "QGAN-LLM (no noise)", 43, "rid", splits, {"data": {"processed_dir": "nope"}},
                                     {"rmse": logged})
    assert path.endswith(os.path.join("EXP-20261107-WINBOX-aaaa", "qgan_llm_no_noise_seed43.npz"))
    loaded = rtv.load_predictions(os.path.dirname(path))
    a = loaded[("qgan_llm_no_noise", 43)]
    assert a["_meta"]["matches_logged_rmse"] is True and len(a["row_index"]) == 50 and a["n_rows_test_shifted"] == len(te.X)
    assert tv.overall_skill(a)["rmse"] == pytest.approx(logged)
    # a wrong logged RMSE is flagged, never raised
    p2 = pd_.dump_eval_predictions(FakeModel(), "QGAN-LLM", 43, "rid", splits, {"data": {}}, {"rmse": logged * 2})
    assert rtv.load_predictions(os.path.dirname(p2))[("qgan_llm", 43)]["_meta"]["matches_logged_rmse"] is False


def test_dump_is_skipped_for_head_only_models_and_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    head_only = types.SimpleNamespace(_gen_path=False)
    assert pd_.dump_eval_predictions(head_only, "X", 1, "r", {}, {}, {}) is None
    monkeypatch.setenv("PREDICTION_DUMP", "0")
    assert pd_.dump_eval_predictions(FakeModel(), "X", 1, "r", {}, {}, {}) is None


def test_end_to_end_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "results" / "predictions" / "EXP-20261107-WINBOX-aaaa"
    os.makedirs(d)
    for seed in range(43, 53):
        base = make_arrs(seed=seed, shared_rows_seed=seed)
        arms = {"qgan_llm": base, "classical_gan_llm": {**base, "y_pred": base["y_true"] * 0.2},
                "classical_gan_llm_matched": {**base, "y_pred": base["y_true"] * 0.25 + np.random.default_rng(seed).normal(0, 2e-5, N_ROWS)}}
        for slug, a in arms.items():
            np.savez_compressed(d / f"{slug}_seed{seed}.npz", meta=np.array(json.dumps(
                {"matches_logged_rmse": True, "n_rows_test_shifted": N_TEST})),
                **{k: v for k, v in a.items() if k != "n_rows_test_shifted"})
    monkeypatch.setattr(sys, "argv", ["x", "--experiment-id", "EXP-20261107-WINBOX-aaaa", "--n-boot", "100"])
    assert rtv.main() == 0
    out = json.load(open(tmp_path / "results" / "temporal_validation__EXP-20261107-WINBOX-aaaa.json"))
    assert set(out["conditions"]) == {"qgan_llm", "classical_gan_llm", "classical_gan_llm_matched"}
    assert out["conditions"]["qgan_llm"]["n_seeds"] == 10
    assert "NOT rolling-origin" in out["design_note"]
    assert set(out["advantage_rule"]) == {"classical_gan_llm", "classical_gan_llm_matched"}
    assert out["advantage_rule"]["classical_gan_llm"]["status"] == "ok"
    printed = capsys.readouterr().out
    assert "pre-registered rule vs classical_gan_llm" in printed


def test_missing_predictions_are_reported_not_guessed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--experiment-id", "EXP-20261107-WINBOX-aaaa"])
    assert rtv.main() == 2 and "no prediction files" in capsys.readouterr().out
