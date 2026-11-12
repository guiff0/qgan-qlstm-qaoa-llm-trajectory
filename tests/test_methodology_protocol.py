"""Pure-numpy tests for the methodology-figure work: the five identical metrics, the naive-baseline CLI helpers, the compliance
checker, the chronological poisoning split + calibration recipe, and the feature-assisted detector prompt."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import run_naive_baseline as nb  # noqa: E402
from src.baselines.feature_map_forecaster import DEFAULT_RIDGE_GRID, FeatureMapForecaster  # noqa: E402
from src.evaluation import arm_metrics as am  # noqa: E402
from src.evaluation import poisoning_protocol as pp  # noqa: E402
from src.evaluation import protocol_compliance as pc  # noqa: E402
from src.evaluation import walk_forward as wf  # noqa: E402
from src.llm import threat_calibration as tc  # noqa: E402


def _data(n=12000, d=6, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    y = 1.1 + np.cumsum(r.normal(size=n) * 1e-4 + 2e-5 * X[:, 0])
    return X, y


def test_every_arm_reports_the_five_metrics():
    X, y = _data()
    for arm in ("linear", "classical_rf"):
        m = FeatureMapForecaster(arm, 4, 2, "ring", 1.0, 0)
        r = m.fit(X[:6000], y[:6000], X[6000:8000], y[6000:8000]).evaluate(X[8000:], y[8000:], attack_rows=60)
        for k in am.METRIC_KEYS:
            assert k in r and np.isfinite(r[k]), (arm, k)
        assert r["map_on_inference_path"] == (arm != "linear")


def test_asr_is_nonzero_for_a_model_that_reacts_to_inputs():
    X = np.random.default_rng(1).normal(size=(500, 3))
    ASR = am.adversarial_asr(lambda Z: 0.01 * Z[:, 0] + 1e-4 * Z[:, 1], X, X.std(0), eps=0.5, n_rows=200)
    assert ASR["asr"] > 0


def test_walk_forward_windows_are_leak_free_and_shared_with_the_baseline():
    X, y = _data(20000)
    W = wf.make_windows(len(y), 3, 3000, min_train=5000)
    wf.assert_no_leak(W)
    a = pc.check_walk_forward({"windows": W, "per_run": [], "args": {}}, None)
    assert next(c for c in a if c["id"] == "M1_chronological_splits")["status"] == pc.PASS


def test_compliance_flags_missing_arm_and_missing_metrics():
    X, y = _data(20000)
    W = wf.make_windows(len(y), 2, 3000, min_train=5000)
    rows = []
    for w in W:
        for seed in (0, 1):
            Xtr, ytr, Xva, yva, Xte, yte = wf.slice_window(X, y, w)
            m = FeatureMapForecaster("classical_rf", 4, 2, "ring", 1.0, seed)
            r = m.fit(Xtr, ytr, Xva, yva).evaluate(Xte, yte, attack_rows=40)
            r.pop("errors"); r.update(seed=seed, window=w["window"]); rows.append(r)
    res = {c["id"]: c["status"] for c in pc.check_walk_forward({"windows": W, "per_run": rows, "args": {}}, None, ridge_grid=DEFAULT_RIDGE_GRID)}
    assert res["M2_three_arms_present"] == pc.FAIL            # Arm 1 and Arm 3 not there
    s = pc.summarise(pc.check_walk_forward({"windows": W, "per_run": rows, "args": {}}, None))
    assert not s["conforms"] and "NOT" in pc.caveat(s)


def test_annotate_verdicts_attaches_methodology_to_every_verdict():
    s = pc.summarise([pc._chk("M2", "x", pc.FAIL, "d")])
    v = pc.annotate_verdicts({"H1": {"why": []}, "H3": {"why": []}}, s)
    assert all("methodology" in d and d["why"] for d in v.values())


def test_naive_baseline_status_and_pairing(tmp_path):
    cells = {(1, 0), (1, 1)}
    own = [{"arm": "linear", "window": 1, "seed": 0, **{k: 1.0 for k in am.METRIC_KEYS}}]
    other = {"per_run": [{"arm": a, "window": 1, "seed": 0, **{k: 2.0 for k in am.METRIC_KEYS}} for a in ("classical_rf", "quantum")]}
    st = nb.status(cells, own, other)
    assert st["linear"]["pending"] == 1 and st["classical_rf"]["pending"] == 1 and not st["others_complete"]
    pr = nb.pairing(own, other, cells)
    assert pr["n_common_cells"] == 1 and pr["arms"]["linear"]["rmse"]["mean"] == 1.0
    p = str(tmp_path / "x" / "b.json"); nb.atomic_write(p, {"a": 1}); assert json.load(open(p)) == {"a": 1}


def test_poisoning_split_is_chronological_and_has_an_embargo():
    sp = pp.chronological_split(1000, 100)
    idx = np.arange(1000)
    assert idx[sp["train"]].max() < idx[sp["eval"]].min() - 0     # strictly earlier
    assert idx[sp["eval"]].min() - idx[sp["train"]].max() >= 2    # embargo row between them
    assert sp["pc"].stop == 100


def test_poisoning_grid_reaches_below_the_old_floor_and_reports_the_split():
    assert min(pp.DEFAULT_GRID) < 0.5
    out = pp.run_protocol(np.random.default_rng(0).normal(size=(400, 4)), np.random.default_rng(1).normal(size=400),
                          np.random.default_rng(2).normal(size=(200, 4)), np.random.default_rng(3).normal(size=200), seed=0)
    assert out["poisoning_split"] == "chronological"


def test_feature_assist_prompt_contains_the_statistic():
    w = [1.1000, 1.1001, 1.1000, 1.1001, 1.1000, 1.1001, 1.1000, 1.1001, 1.1000, 1.1030]
    line = tc.window_stats_line(w)
    assert "x the SD" in line and "computed" in line
    assert "x the SD" in tc.few_shot_block([(w, True)], assist=True) and "x the SD" not in tc.few_shot_block([(w, True)])
    z_attack, z_clean = tc.last_jump_z(w), tc.last_jump_z(w[:-1] + [1.1001])
    assert z_attack > 3 > z_clean


def test_apply_calibration_writes_fraction():
    from scripts.run_study import apply_calibration, poisoning_section
    cfg = "poisoning:\n  protocol: v2\n  fraction: 0.2\n  shift_sd: 0.5\n  calibrated: false\n"
    new = apply_calibration(cfg, 0.3, 0.4)
    assert "fraction: 0.4" in new and poisoning_section(new)["calibrated"] and poisoning_section(new)["shift_sd"] == 0.3
