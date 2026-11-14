"""14 Nov 2026: Classical LSTM and QAOA-Enhanced LLM forecast the next-close CHANGE (forecast_target: delta, the default), so
their pilot-table rows are measured the same way as the other pilots. `forecast_target: level` keeps the October behaviour."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.baselines import classical_lstm as cl      # noqa: E402
from src.baselines import qaoa_llm as ql            # noqa: E402

import os       # noqa: E402
import yaml     # noqa: E402

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "default_config.yaml"), encoding="utf8") as _fh:
    ATT = {**yaml.safe_load(_fh)["adversarial"], "attacks": ["fgsm"]}      # the study's own attack settings, FGSM only (fast)


def _series(n=900, k=6, seed=0, shift=0.0):
    """Random-walk close around 1.1 (+ shift) with stationary features; one feature carries next-bar information."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k)).astype(np.float32)
    step = 1e-4 * (0.6 * np.roll(X[:, 0], 1) + rng.normal(size=n))      # change into row j depends on feature 0 at row j-1
    y = 1.1 + shift + np.cumsum(step)
    return X, y


def _lstm(tmp_path, frame, **kw):
    m = cl.ClassicalLSTM({"input_size": 6, "hidden_size": 8, "num_layers": 1, "sequence_length": 5, "epochs": 2, "batch_size": 32,
                          "max_batches_per_epoch": 10, "val_samples": 64, "forecast_target": frame, **kw}, seed=1)
    m.checkpoint_path = str(tmp_path / f"lstm_{frame}.pt")
    return m


def test_window_targets_delta_is_the_change_after_the_window_and_level_is_unchanged():
    y = np.array([1.0, 1.5, 1.2, 2.0])
    assert np.allclose(cl.window_targets(y, "level"), y)
    assert np.allclose(cl.window_targets(y, "delta", scale=0.5), [0.0, 1.0, -0.6, 1.6])
    big = 1.1 + np.arange(5) * 1e-5                                   # float32 differencing would lose these
    assert np.allclose(cl.window_targets(big, "delta", scale=1e-5)[1:], 1.0, atol=1e-6)


def test_lstm_default_is_delta_with_its_own_checkpoint_file():
    d, lv = cl.ClassicalLSTM({}), cl.ClassicalLSTM({"forecast_target": "level"})
    assert d.target_frame == "delta" and d.checkpoint_path == cl.CHECKPOINT_PATH_DELTA
    assert lv.target_frame == "level" and lv.checkpoint_path == cl.CHECKPOINT_PATH and cl.CHECKPOINT_PATH != cl.CHECKPOINT_PATH_DELTA


def test_lstm_that_outputs_zero_scores_exactly_persistence_in_the_delta_frame(tmp_path):
    X, y = _series()
    m = _lstm(tmp_path, "delta")
    m.build()
    with torch.no_grad():
        m.fc.weight.zero_(); m.fc.bias.zero_()
    m.target_scale = 1e-4
    r = m.evaluate(X, y)
    assert r["target_frame"] == "delta" and r["rmse"] == pytest.approx(r["persistence_rmse"], rel=1e-9)
    assert r["persistence_rmse"] == pytest.approx(float(np.sqrt(np.mean(np.diff(y)[4:] ** 2))), rel=1e-9)


def test_lstm_delta_training_survives_a_price_level_shift_and_level_does_not(tmp_path):
    Xtr, ytr = _series(seed=0)
    Xva, yva = _series(n=300, seed=1)
    Xte, yte = _series(n=400, seed=2, shift=0.05)                     # test prices sit 0.05 above the training prices
    out = {}
    for frame in ("delta", "level"):
        m = _lstm(tmp_path, frame)
        m.train(Xtr, ytr, Xva, yva)
        out[frame] = m.evaluate(Xte, yte, attack_cfg=ATT, last_input_prices=yte)
        assert out[frame]["target_frame"] == frame and np.isfinite(out[frame]["asr"])
    d, lv = (out[f]["rmse"] / out[f]["persistence_rmse"] for f in ("delta", "level"))
    assert d < 3.0 and lv > 50.0, (d, lv)
    assert 0 < _lstm(tmp_path, "delta").target_scale == 1.0           # scale is learned in train(), not at construction


def test_lstm_checkpoint_records_the_frame_and_refuses_the_other_one(tmp_path):
    Xtr, ytr = _series()
    m = _lstm(tmp_path, "delta")
    m.train(Xtr, ytr, Xtr[:200], ytr[:200])
    ck = torch.load(m.checkpoint_path, map_location="cpu")
    assert ck["target_frame"] == "delta" and ck["target_scale"] == pytest.approx(float(np.std(np.diff(ytr))), rel=1e-6)
    again = _lstm(tmp_path, "delta"); again.build(); again.load_model(m.checkpoint_path)
    assert again.target_scale == pytest.approx(m.target_scale)
    other = _lstm(tmp_path, "level"); other.build()
    with pytest.raises(RuntimeError, match="trained in the 'delta' frame"):
        other.load_model(m.checkpoint_path)
    legacy = tmp_path / "legacy.pt"                                    # an October checkpoint has no frame: it is level
    torch.save({"format_version": 2, "lstm": m.model.state_dict(), "fc": m.fc.state_dict()}, legacy)
    with pytest.raises(RuntimeError, match="trained in the 'level' frame"):
        again.load_model(str(legacy))
    other.load_model(str(legacy))


class _FakeSelection:
    def __init__(self, k, picked):
        self.selected_mask = np.zeros(k, dtype=bool); self.selected_mask[list(picked)] = True
        self.qubo_objective = -0.5


def _qaoa(monkeypatch, frame, seen, **kw):
    m = ql.QAOALLM({"n_features": 6, "epochs": 3, "batch_size": 32, "learning_rate": 0.05, "forecast_target": frame, **kw}, seed=1)

    def fake(X, y, run_logger=None):                                   # no PennyLane needed: the selector is replaced
        seen["X"], seen["y"] = np.asarray(X), np.asarray(y)
        return _FakeSelection(6, (0, 3))
    monkeypatch.setattr(m, "_select_features", fake)
    return m


def test_qaoa_enhanced_delta_selects_and_trains_on_the_next_bar_change(monkeypatch):
    Xtr, ytr = _series(seed=0)
    Xte, yte = _series(n=400, seed=2, shift=0.05)
    seen = {}
    m = _qaoa(monkeypatch, "delta", seen, feature_frame="level", head_fit="sgd")     # the 14 Nov morning conversion
    m.train(Xtr, ytr, None, None)
    assert seen["X"].shape == (len(Xtr) - 1, 6) and np.allclose(seen["y"] * m.target_scale, np.diff(ytr), atol=1e-12)
    assert np.std(seen["y"]) == pytest.approx(1.0, rel=1e-6)           # standardised target reaches the selector and the head
    r = m.evaluate(Xte, yte, attack_cfg=ATT, last_input_prices=yte)
    assert r["target_frame"] == "delta" and r["persistence_rmse"] == pytest.approx(float(np.sqrt(np.mean(np.diff(yte) ** 2))), rel=1e-9)
    assert r["rmse"] / r["persistence_rmse"] < 1.0                     # feature 0 carries next-bar information here
    assert np.isfinite(r["asr"])
    with torch.no_grad():
        m.forecast_head.weight.zero_(); m.forecast_head.bias.zero_()
    z = m.evaluate(Xte, yte)
    assert z["rmse"] == pytest.approx(z["persistence_rmse"], rel=1e-9)  # a head that learns nothing = persistence


def test_qaoa_enhanced_level_frame_is_the_october_behaviour(monkeypatch):
    Xtr, ytr = _series(seed=0)
    seen = {}
    m = _qaoa(monkeypatch, "level", seen)
    m.train(Xtr, ytr, None, None)
    assert seen["X"].shape == Xtr.shape and np.array_equal(seen["y"], ytr) and m.target_scale == 1.0
    r = m.evaluate(Xtr, ytr)
    assert r["target_frame"] == "level" and "persistence_rmse" not in r
    assert r["rmse"] == pytest.approx(float(np.sqrt(np.mean((ytr - m.predict(Xtr).flatten()) ** 2))), rel=1e-6)


# ---- 14 Nov (afternoon): drift-free inputs, no-change start, closed-form head -------------------------------------------

def test_qaoa_enhanced_defaults_are_diff_inputs_and_a_ridge_head_with_no_look_ahead(monkeypatch):
    Xtr, ytr = _series(seed=0)
    seen = {}
    m = _qaoa(monkeypatch, "delta", seen, ridge=1.0)
    assert (m.feature_frame, m.head_fit) == ("diff", "ridge")
    m.train(Xtr, ytr, None, None)
    # row j: inputs X[j+1] - X[j] (known at bar j+1)  ->  change y[j+2] - y[j+1]
    assert np.allclose(seen["X"], (Xtr[1:] - Xtr[:-1])[:-1]) and np.allclose(seen["y"] * m.target_scale, ytr[2:] - ytr[1:-1], atol=1e-12)
    # the closed-form head is the least-squares solution on the selected columns (ridge 1.0 is negligible at n = 898)
    A = np.c_[seen["X"][:, [0, 3]].astype(np.float64), np.ones(len(seen["X"]))]
    w = np.linalg.lstsq(A, seen["y"], rcond=None)[0]
    assert np.allclose(m.forecast_head.weight.detach().numpy().ravel(), w[:2], atol=2e-3)
    assert float(m.forecast_head.bias) == pytest.approx(w[2], abs=2e-3)
    Xte, yte = _series(n=400, seed=2, shift=0.05)
    r = m.evaluate(Xte, yte, attack_cfg=ATT, last_input_prices=yte)
    assert r["feature_frame"] == "diff" and r["qaoa_n_selected"] == 2 and np.isfinite(r["asr"])
    assert r["persistence_rmse"] == pytest.approx(float(np.sqrt(np.mean(np.diff(yte)[1:] ** 2))), rel=1e-9)
    # a constant added to every feature (a level shift between training and test) cannot change a diff-input forecast
    r2 = m.evaluate(Xte + 7.0, yte)
    assert r2["rmse"] == pytest.approx(r["rmse"], rel=1e-5)


def test_qaoa_enhanced_rejects_diff_inputs_in_the_level_frame():
    with pytest.raises(ValueError, match="needs forecast_target: delta"):
        ql.QAOALLM({"forecast_target": "level", "feature_frame": "diff"})


def test_lstm_window_centering_makes_the_forecast_blind_to_the_feature_level(tmp_path):
    X, y = _series(n=300)
    m = _lstm(tmp_path, "delta")
    assert m.input_centering == "window" and _lstm(tmp_path, "level").input_centering == "none"
    m.build()
    with torch.no_grad():
        m.fc.weight.normal_(); m.target_scale = 1e-4
    a, b = m.predict(X), m.predict(X + 5.0)
    assert np.allclose(a, b, atol=1e-9)
    off = _lstm(tmp_path, "delta", input_centering="none"); off.build()
    with torch.no_grad():
        off.fc.weight.normal_()
    assert not np.allclose(off.predict(X), off.predict(X + 5.0), atol=1e-6)


def test_lstm_starts_at_no_change_and_keeps_it_when_no_epoch_beats_it_on_validation(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(700, 6)).astype(np.float32)
    y = 1.1 + np.cumsum(1e-4 * rng.normal(size=700))                 # pure random walk: nothing to learn
    m = _lstm(tmp_path, "delta", learning_rate=0.05, epochs=3, early_stopping_patience=3)
    m.train(X[:500], y[:500], X[500:], y[500:])
    ck = torch.load(m.checkpoint_path, map_location="cpu")
    r = m.evaluate(X[500:], y[500:])
    assert ck["input_centering"] == "window" and r["best_epoch"] in (-1, 0, 1, 2)
    if r["best_epoch"] == -1:                                        # the kept model is the untouched no-change forecast
        assert r["rmse"] == pytest.approx(r["persistence_rmse"], rel=1e-9)
    assert r["rmse"] / r["persistence_rmse"] < 1.5                   # never the 38x of the level frame


def test_qaoa_ridge_auto_is_chosen_on_validation_and_falls_back_to_no_change(monkeypatch):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(1500, 6)).astype(np.float32)
    y = 1.1 + np.cumsum(1e-4 * rng.normal(size=1500))                # pure random walk: the features carry nothing
    seen = {}
    m = _qaoa(monkeypatch, "delta", seen)
    assert m.config["ridge"] == "auto"
    m.train(X[:1000], y[:1000], X[1000:], y[1000:])
    hi = m.head_info
    assert hi["mode"].startswith("auto") and hi["val_rmse_over_no_change"] <= 1.0 and "no_change" in hi["candidates_val_rmse_over_no_change"]
    if hi["no_change"]:
        r = m.evaluate(X[1000:], y[1000:])
        assert r["head_is_no_change"] is True and r["rmse"] == pytest.approx(r["persistence_rmse"], rel=1e-9)
    # with real structure the search keeps a head and it beats no-change on validation
    Xs, ys = _series(n=1500, seed=3)
    m2 = _qaoa(monkeypatch, "delta", {})
    m2.train(Xs[:1000], ys[:1000], Xs[1000:], ys[1000:])
    assert m2.head_info["no_change"] is False and m2.head_info["val_rmse_over_no_change"] < 0.99
    assert m2.evaluate(Xs[1000:], ys[1000:])["head_is_no_change"] is False
    # no validation rows: the old fixed penalty
    m3 = _qaoa(monkeypatch, "delta", {}); m3.train(Xs[:1000], ys[:1000], None, None)
    assert m3.head_info == {"mode": "fixed", "penalty": 1.0, "no_change": False}
