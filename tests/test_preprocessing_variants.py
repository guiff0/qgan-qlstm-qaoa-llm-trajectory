import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import compare_sensitivity as cs  # noqa: E402
from scripts import run_sensitivity_study as rs  # noqa: E402
from scripts.prepare_data import expand_feature_space  # noqa: E402
from src.data import variants as dv  # noqa: E402
from src.utils import experiment as exp  # noqa: E402

CFG = {"train_end": "2021-12-31"}


def frame(n=60_000, spike_in="train"):
    """Minute bars. Large enough that a 99.9th percentile is not decided by the single corrupt bar (the real
    training split has millions of rows); the 'test' variant starts late so most rows fall after train_end."""
    rng = np.random.default_rng(0)
    start = "2021-06-01" if spike_in == "train" else "2021-12-20"
    ts = pd.date_range(start, periods=n, freq="min")
    close = 1.1 + np.cumsum(rng.normal(0, 1e-3, n))
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": close + 1e-3, "low": close - 1e-3, "close": close,
                       "volume": rng.integers(100, 1000, n).astype(float)})
    i = 100 if spike_in == "train" else n - 10
    df.loc[i, "volume"] = 446_000_000.0
    return df


def test_baseline_changes_nothing():
    df = frame()
    before = df.copy()
    out, info = dv.pre_features(df, "baseline", CFG)
    assert info == {} and out.equals(before)
    out2, cols = dv.post_features(out.copy(), ["close"], "baseline")
    assert cols == ["close"] and list(out2.columns) == list(before.columns)
    assert not dv.skips_indicator_repair("baseline") and dv.skips_indicator_repair("no_interpolation")


def test_winsor_clips_the_corrupt_bar_using_training_rows_only():
    df, info = dv.pre_features(frame(spike_in="train"), "volume_winsor", CFG)
    assert df["volume"].max() < 10_000 and info["n_clipped"] >= 1 and info["max_volume_before"] == 446_000_000.0
    assert info["fitted_on"] == "training period only"
    # a spike that lies AFTER train_end must not influence the cap
    df2, info2 = dv.pre_features(frame(spike_in="test"), "volume_winsor", CFG)
    train_only = frame(spike_in="test")
    train_only = train_only[train_only["timestamp"] <= "2021-12-31"]
    assert 0 < len(train_only) < len(frame(spike_in="test"))
    assert info2["volume_cap"] == pytest.approx(float(train_only["volume"].quantile(dv.WINSOR_QUANTILE)))
    assert df2["volume"].max() == pytest.approx(info2["volume_cap"])        # ... but it is clipped too


def test_documented_features_add_the_rolling_moments_and_keep_the_baseline_ones():
    df, base_cols = expand_feature_space(frame())
    n_base = len(base_cols)
    df2, cols = dv.post_features(df.copy(), base_cols, "documented_features")
    new = [c for c in cols if c.startswith("roll_")]
    assert len(cols) == n_base + 8 and set(new) == {f"roll_{k}_{w}" for k in ("skew", "kurt") for w in dv.DOC_WINDOWS}
    assert all(c in df2.columns for c in new) and np.isfinite(df2[new].dropna().to_numpy()).all()


def test_robust_scaler_is_resistant_to_a_spike_and_invertible():
    rng = np.random.default_rng(1)
    X = rng.normal(5, 2, (500, 3))
    X[0, 0] = 1e9
    rob, std = dv.fit_variant_scaler("robust_scaler", X), dv.fit_variant_scaler("baseline", X)
    assert abs(np.median(rob.transform(X)[1:, 0])) < 0.2 and std.transform(X)[1:, 0].std() < 0.1      # std scaler is flattened
    assert np.allclose(rob.inverse_transform(rob.transform(X[:5])), X[:5], rtol=1e-6)


def test_full_feature_pipeline_with_the_corrupt_bar(tmp_path):
    base, _ = expand_feature_space(frame(20_000))
    win_df, _ = dv.pre_features(frame(20_000), "volume_winsor", CFG)
    win, _ = expand_feature_space(win_df)
    assert base["volume"].max() > 1e8 and win["volume"].max() < 1e4
    assert base["volume_sma_10"].max() > 1e7 and win["volume_sma_10"].max() < 1e4          # volume features no longer dominated


def test_manifest_and_default_dirs(tmp_path):
    p = dv.write_manifest(str(tmp_path / "v"), "volume_winsor", {"cap": 1}, ["a", "b"], {"train": 10})
    rec = json.load(open(p))
    assert rec["variant"] == "volume_winsor" and rec["n_features_raw"] == 2
    assert dv.default_processed_dir("volume_winsor").endswith(os.path.join("processed_variants", "volume_winsor"))
    with pytest.raises(ValueError):
        dv.check_variant("typo")


def test_variant_configs_differ_in_data_and_checkpoint_identity(tmp_path):
    base = yaml.safe_load(open(os.path.join(ROOT, "config", "default_config.yaml"), encoding="utf-8"))
    a, b = rs.variant_config(base, "volume_winsor"), rs.variant_config(base, "robust_scaler")
    assert a["data"]["processed_dir"] != base["data"]["processed_dir"] != b["data"]["processed_dir"]
    for sec in ("qgan_llm", "classical_gan_llm"):
        assert a[sec]["data_variant"] == "volume_winsor" and a[sec] != b[sec] and "data_variant" not in base[sec]
    assert base["data"]["processed_dir"] == yaml.safe_load(open(os.path.join(ROOT, "config", "default_config.yaml"), encoding="utf-8"))["data"]["processed_dir"]
    paths = rs.write_variant_configs(os.path.join(ROOT, "config", "default_config.yaml"), str(tmp_path))
    assert set(paths) == set(dv.VARIANTS) and all(os.path.isfile(p) for p in paths.values())


def test_plan_never_points_a_variant_at_the_main_processed_folder():
    items = rs.plan("EXP-20261107-WINBOX-aaaa")
    assert {i["variant"] for i in items} == set(dv.VARIANTS)
    for i in items:
        assert f"--processed-dir {i['processed_dir']}" in i["commands"][0]          # also for 'baseline'
        assert "--parent-experiment EXP-20261107-WINBOX-aaaa" in i["commands"][1] and "--new-experiment" in i["commands"][1]


def _rows(slug, eid, fids, base_ts):
    return [{"run_id": f"{slug}_{base_ts}{i:02d}Z", "seed": 43 + i, "experiment_id": eid, "fid": f,
             "rmse": 1e-4, "persistence_rmse": 1e-4, "asr": 30.0} for i, f in enumerate(fids)]


def test_compare_sensitivity_flags_a_changed_conclusion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parent = "EXP-20261107-WINBOX-aaaa"
    cond = ["QGAN-LLM (12-qubit)", "Classical GAN-LLM (matched)"]
    ids = {"baseline": "EXP-20261108-WINBOX-b001", "volume_winsor": "EXP-20261108-WINBOX-b002", "robust_scaler": "EXP-20261108-WINBOX-b003"}
    rng = np.random.default_rng(0)
    rows = []
    qgan_lower = {"baseline": 100, "volume_winsor": 120, "robust_scaler": 100}          # winsor: the advantage vanishes
    for v, eid in ids.items():
        exp.create_experiment(eid, conditions=cond, seeds=range(43, 48), base_seed=43, label=f"sensitivity:{v}", parent=parent)
        rows += _rows("qgan_llm_12_qubit", eid, rng.normal(qgan_lower[v], 2, 5), "20261108T1000")
        rows += _rows("classical_gan_llm_matched", eid, rng.normal(120, 2, 5), "20261108T1100")
    pd.DataFrame(rows).to_csv(tmp_path / "all.csv", index=False)
    rep = cs.compare(parent, str(tmp_path / "all.csv"))
    assert set(rep["variants"]) == set(ids) and rep["baseline_present"]
    assert rep["changes"]["volume_winsor"]["fid"]["changed"] is True and rep["changes"]["robust_scaler"]["fid"]["changed"] is False
    assert rep["conclusions_robust"] is False
