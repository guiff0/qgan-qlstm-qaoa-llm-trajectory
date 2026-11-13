"""Three-arm walk-forward + naive baseline: every iteration (window x seed) must use the same rows and the same protocol, stale
results must not be reused, and the hypothesis tests must find the baseline file. Pure numpy (no torch / PennyLane)."""
import json
import os

import numpy as np
import pytest

from src.evaluation import protocol_compliance as pc
from src.evaluation import walk_forward as wf


class _Rec:
    calls = []

    def __init__(self, arm, *a, **k):
        self.arm = arm

    def fit(self, *a):
        return self

    def evaluate(self, X, y, **kw):
        _Rec.calls.append((self.arm, kw))
        return {"arm": self.arm, "skill": 0.0, "rmse": 1.0, "persistence_rmse": 1.0, "errors": np.zeros(len(X) - 1)}


def test_every_arm_of_a_cell_gets_the_same_attack_rows_and_metric_seed():
    _Rec.calls = []
    X, y = np.zeros((900, 4)), np.arange(900.0)
    w = wf.make_windows(900, 2, 200, min_train=200, val_frac=0.2)
    wf.run_windows(X, y, w, [0, 3, 7], forecaster=_Rec, dm_fn=lambda a, b: {"statistic": 0.0}, log=lambda *_: None,
                   attack_rows=123, attack_eps=0.07)
    assert len(_Rec.calls) == 2 * 3 * 3
    seeds_seen = [kw["metric_seed"] for _, kw in _Rec.calls]
    assert seeds_seen == [s for _ in range(2) for s in (0, 0, 0, 3, 3, 3, 7, 7, 7)]       # seed of the cell, never a fixed 0
    assert {(kw["attack_rows"], kw["attack_eps"]) for _, kw in _Rec.calls} == {(123, 0.07)}


def _row(arm, win, seed, **kw):
    r = {"arm": arm, "window": win, "seed": seed, "skill": 0.0, "rmse": 1.0, "asr": 1.0, "fidelity": 0.5, "latency_ms": 0.1,
         "n_test": 100, "asr_n": 300, "asr_eps": 0.1, "metric_seed": seed, "n_features": {"linear": 6}.get(arm, 10),
         "map_on_inference_path": arm != "linear"}
    r.update(kw)
    return r


def _files(naive_seed_override=None):
    arms = ("linear", "classical_rf", "quantum")
    wins = wf.make_windows(2000, 2, 300, min_train=500)
    rows = [_row(a, w["window"], s) for a in arms for w in wins for s in (0, 1)]
    naive = {"windows": wins, "per_run": [r if naive_seed_override is None or r["arm"] != "linear"
                                          else {**r, "metric_seed": naive_seed_override} for r in rows if r["arm"] == "linear"]}
    return {"windows": wins, "args": {"n_qubits": 4, "n_layers": 2}, "per_run": [r for r in rows if r["arm"] != "linear"]}, naive


def _status(checks, cid):
    return next(c for c in checks if c["id"] == cid)["status"]


def test_per_cell_protocol_check_passes_when_identical_and_fails_when_arm1_uses_other_rows():
    wfj, nv = _files()
    assert _status(pc.check_walk_forward(wfj, nv), "M4b_per_cell_identical_protocol") == pc.PASS
    wfj, nv = _files(naive_seed_override=0)                       # the old behaviour: Arm 1 scored with metric_seed 0 for every seed
    chk = pc.check_walk_forward(wfj, nv)
    assert _status(chk, "M4b_per_cell_identical_protocol") == pc.FAIL


def test_per_cell_check_is_not_assessable_for_rows_without_metric_seed():
    wfj, nv = _files()
    for r in wfj["per_run"] + nv["per_run"]:
        r.pop("metric_seed")
    assert _status(pc.check_walk_forward(wfj, nv), "M4b_per_cell_identical_protocol") == pc.NA


def test_find_naive_falls_back_to_the_shared_legacy_file_but_prefers_the_experiments_own(tmp_path):
    root = str(tmp_path)
    assert pc.find_naive("EXP-1", root) is None
    os.makedirs(tmp_path / "naive_baseline" / "legacy")
    (tmp_path / "naive_baseline" / "legacy" / "naive_baseline.json").write_text(json.dumps({"per_run": [], "who": "legacy"}))
    assert pc.find_naive("EXP-1", root)["who"] == "legacy"
    os.makedirs(tmp_path / "naive_baseline" / "EXP-1")
    (tmp_path / "naive_baseline" / "EXP-1" / "naive_baseline.json").write_text(json.dumps({"per_run": [], "who": "own"}))
    assert pc.find_naive("EXP-1", root)["who"] == "own"


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    n = 4200
    X = rng.normal(size=(n, 6))
    y = 1.1 + np.cumsum(rng.normal(0, 1e-4, n))
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    for s, (a, b) in {"train": (0, 2600), "val": (2600, 3400), "test": (3400, n)}.items():
        np.save(d / f"X_{s}.npy", X[a:b])
        np.save(d / f"y_{s}.npy", y[a:b])
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_end_to_end_cells_match_and_stale_baseline_is_refitted(tiny_data):
    import scripts.run_naive_baseline as nb
    import scripts.run_walk_forward as rwf
    a = ["--windows", "2", "--test-len", "400", "--min-train", "800", "--seeds", "0", "1", "--n-qubits", "4", "--n-layers", "2",
         "--out", "results/walk_forward_q4.json"]
    assert rwf.main(a + ["--skip-if-current"]) == 0
    first = os.path.getmtime("results/walk_forward_q4.json")
    assert rwf.main(a + ["--skip-if-current"]) == 0 and os.path.getmtime("results/walk_forward_q4.json") == first   # complete: skipped
    assert nb.main(["--like", "results/walk_forward_q4.json", "--experiment-id", "legacy"]) == 0
    w = json.load(open("results/walk_forward_q4.json"))
    n = json.load(open("results/naive_baseline/legacy/naive_baseline.json"))
    checks = pc.check_walk_forward(w, pc.find_naive("EXP-new", "results"))
    assert _status(checks, "M4b_per_cell_identical_protocol") == pc.PASS
    assert _status(checks, "M1_chronological_splits") == pc.PASS and _status(checks, "M5_paired_seeds_windows") == pc.PASS
    lin = {(r["window"], r["seed"]): r for r in w["per_run"] if r["arm"] == "linear"}
    for r in n["per_run"]:                                      # Arm 1 in both files is the same fit on the same cell
        o = lin[(r["window"], r["seed"])]
        assert o["rmse"] == r["rmse"] and o["asr"] == r["asr"] and o["metric_seed"] == r["metric_seed"] == r["seed"]
    # a file from an older ridge grid is not reused
    n["ridge_grid"] = [1e-8, 1.0]
    json.dump(n, open("results/naive_baseline/legacy/naive_baseline.json", "w"))
    assert nb.main(["--like", "results/walk_forward_q4.json", "--experiment-id", "legacy"]) == 0
    assert any(f.startswith("naive_baseline.json.stale_") for f in os.listdir("results/naive_baseline/legacy"))
    # a walk-forward file from an older grid is refitted, not skipped
    w["ridge_grid"] = [1e-8, 1.0]
    json.dump(w, open("results/walk_forward_q4.json", "w"))
    assert rwf.main(a + ["--skip-if-current"]) == 0
    assert json.load(open("results/walk_forward_q4.json"))["ridge_grid"][0] == 1e-10


# ---- M6: the ridge optimum on the grid edge is followed outward; the null model is a finding, not under-tuning -------------------
def _fit(kind, arm="linear", seed=0):
    from src.baselines.feature_map_forecaster import FeatureMapForecaster
    rng = np.random.default_rng(seed)
    n = 3000
    X = rng.normal(size=(n, 6))
    y = 1.1 + np.cumsum(rng.normal(0, 1e-4, n) if kind == "noise" else 1e-4 * X[:, 0] + rng.normal(0, 1e-6, n))
    m = FeatureMapForecaster(arm, 4, 2, "ring", 1.0, seed).fit(X[:1800], y[:1800], X[1800:2400], y[1800:2400])
    return m.evaluate(X[2400:], y[2400:], full_metrics=False)


def test_pure_noise_selects_the_null_model_and_reports_it_instead_of_an_edge_failure():
    r = _fit("noise")
    assert r["ridge_status"] == "saturated_null" and r["ridge_extended"] and r["ridge"] > 1e6
    assert abs(r["val_mse"] - r["val_mse_null"]) <= 1e-3 * r["val_mse_null"]
    assert abs(r["skill"]) < 0.01                                   # equals persistence, not "better than"


def test_real_signal_gives_an_interior_optimum_and_positive_skill():
    r = _fit("signal")
    assert r["ridge_status"] == "interior" and not r["ridge_extended"] and r["skill"] > 0


def test_audit_fails_only_for_unresolved_edges_and_explains_the_null_model():
    from src.evaluation import three_arm
    rows = []
    for arm in ("linear", "classical_rf", "quantum"):
        for s in range(10):
            rows.append({"arm": arm, "seed": s, "skill": 0.0, "rmse": 1.0, "persistence_rmse": 1.0, "n_features": 10, "n_test": 100,
                         "ridge": 1e8 if arm == "linear" else 1.0, "ridge_status": "saturated_null" if arm == "linear" else "interior"})
    au = three_arm.audit(rows, {"n_qubits": 8, "n_layers": 3}, (1e-10, 1e6))
    assert au["all_passed"] and any("null model" in n for n in au["notes"])
    rows[0]["ridge_status"] = "edge_unresolved"
    assert not three_arm.audit(rows, {"n_qubits": 8, "n_layers": 3}, (1e-10, 1e6))["all_passed"]
    assert "null model" in " ".join(three_arm.verdict(rows, [])["notes"])
    for r in rows:
        r["ridge_status"] = "saturated_null"
    dm = [{"seed": s, "vs": c, "statistic": 0.0, "p_value": 0.9} for s in range(10) for c in ("linear", "classical_rf")]
    v = three_arm.verdict(rows, dm)
    assert v["verdict"] == "NOT_DECIDABLE" and "vacuous" not in v["verdict"] and any("same persistence forecast" in n for n in v["notes"])


def test_feature_map_experiment_is_skipped_only_when_complete_and_current(tiny_data):
    import scripts.run_feature_map_experiment as fm
    a = ["--seeds", "0", "1", "--n-qubits", "4", "--n-layers", "2", "--max-train-rows", "1500", "--max-eval-rows", "600",
         "--out", "results/fm.json"]
    assert fm.main_argv(a) == 0
    first = os.path.getmtime("results/fm.json")
    assert fm.main_argv(a + ["--skip-if-current"]) == 0 and os.path.getmtime("results/fm.json") == first
    d = json.load(open("results/fm.json"))
    assert all(r["metric_seed"] == r["seed"] and "ridge_status" in r for r in d["per_run"])
    d["ridge_grid"] = [1e-8]                                        # an older grid: must be refitted
    json.dump(d, open("results/fm.json", "w"))
    assert fm.main_argv(a + ["--skip-if-current"]) == 0
    assert json.load(open("results/fm.json"))["ridge_grid"][0] == 1e-10


# ---- independence: the three-arm path shares no component with either GAN-LLM arm, the LLM, or QAOA ----------------------------------
FORBIDDEN = ("torch", "pennylane", "requests", "src.llm", "src.quantum.qaoa", "src.quantum.circuits", "src.baselines.qgan_llm",
             "src.baselines.classical_gan_llm", "src.baselines.qaoa_llm", "src.baselines.context_llm", "src.baselines.llm_forecaster",
             "src.baselines.qlstm_forecaster", "src.baselines.classical_lstm", "src.attacks", "src.utils.gan_training",
             "src.utils.generator_path")


def test_the_three_scripts_import_no_generator_llm_qaoa_or_torch_even_when_they_run(tiny_data):
    import sys
    import scripts.run_feature_map_experiment as fm
    import scripts.run_naive_baseline as nb
    import scripts.run_walk_forward as rwf
    before = set(sys.modules)
    assert rwf.main(["--windows", "2", "--test-len", "400", "--min-train", "800", "--seeds", "0", "--n-qubits", "4", "--n-layers", "2",
                     "--out", "results/w.json"]) == 0
    assert nb.main(["--like", "results/w.json", "--experiment-id", "legacy"]) == 0
    assert fm.main(["--seeds", "0", "--n-qubits", "4", "--n-layers", "2", "--max-train-rows", "1200", "--max-eval-rows", "500",
                    "--out", "results/f.json"]) == 0
    new = {m for m in set(sys.modules) - before}
    bad = sorted(m for m in new if any(m == f or m.startswith(f + ".") for f in FORBIDDEN))
    assert not bad, f"the three-arm path pulled in generator/LLM/QAOA code: {bad}"


# ---- two widths: a 12-qubit walk-forward is paired only with its own 12-qubit Arm 1 (width 4 stands in for 12 here: same code path) ----
def test_non_default_width_uses_its_own_naive_folder_and_a_mismatch_is_caught(tiny_data):
    import scripts.run_naive_baseline as nb
    import scripts.run_walk_forward as rwf
    a = ["--windows", "2", "--test-len", "400", "--min-train", "800", "--seeds", "0", "1", "--n-layers", "2"]
    assert rwf.main(a + ["--n-qubits", "4", "--out", "results/walk_forward_q4.json"]) == 0
    assert rwf.main(a + ["--n-qubits", "6", "--out", "results/walk_forward_q6.json"]) == 0
    assert nb.main(["--like", "results/walk_forward_q4.json", "--n-qubits", "4", "--n-layers", "2", "--experiment-id", "legacy_q4"]) == 0
    assert nb.main(["--like", "results/walk_forward_q6.json", "--n-qubits", "6", "--n-layers", "2", "--experiment-id", "legacy_q6"]) == 0
    w4, w6 = json.load(open("results/walk_forward_q4.json")), json.load(open("results/walk_forward_q6.json"))
    n4, n6 = pc.find_naive("EXP-x", "results", n_qubits=4), pc.find_naive("EXP-x", "results", n_qubits=6)
    assert n4["found_in"] == "legacy_q4" and n6["found_in"] == "legacy_q6" and n4["args"]["n_qubits"] == 4 and n6["args"]["n_qubits"] == 6
    assert _status(pc.check_walk_forward(w4, n4), "M4c_same_width_depth_grid") == pc.PASS
    assert _status(pc.check_walk_forward(w6, n6), "M4c_same_width_depth_grid") == pc.PASS
    assert _status(pc.check_walk_forward(w6, n4), "M4c_same_width_depth_grid") == pc.FAIL      # 6-qubit arms vs a 4-qubit Arm 1
    assert pc.find_naive("EXP-x", "results") is None                                          # width 8 does not fall back to another width


def test_q1_result_gets_the_methodology_of_its_own_width():
    ok = pc.summarise([pc._chk("M1", "x", pc.PASS, "d")])
    bad = pc.summarise([pc._chk("M2", "x", pc.FAIL, "d")])
    q1 = pc.annotate_three_arm({"8_qubits (a.json)": {"verdict": "X", "notes": []}, "12_qubits (b.json)": {"verdict": "Y", "notes": []}},
                               ok, {8: ok, 12: bad})
    assert q1["8_qubits (a.json)"]["methodology"]["conforms"] and q1["12_qubits (b.json)"]["methodology"]["failed"] == ["M2"]


# ---- circuit-evaluation cache: train rows go through the circuit once in fit(), and nothing numerical changes ---------------------------
def _fit_counting(arm, cache, seed=0):
    from src.baselines.feature_map_forecaster import FeatureMapForecaster
    rng = np.random.default_rng(3)
    n = 2400
    X = rng.normal(size=(n, 6))
    y = 1.1 + np.cumsum(1e-4 * X[:, 0] + rng.normal(0, 1e-5, n))
    m = FeatureMapForecaster(arm, 4, 2, "ring", 1.0, seed, chunk=500)
    m.cache_design = cache
    calls = {"rows": 0}
    orig = m.fmap.transform
    m.fmap.transform = lambda A: (calls.__setitem__("rows", calls["rows"] + len(A)), orig(A))[1]
    m.fit(X[:1400], y[:1400], X[1400:1900], y[1400:1900])
    return m, calls["rows"]


@pytest.mark.parametrize("arm", ["quantum", "classical_rf"])
def test_design_cache_is_bit_identical_and_halves_the_train_circuit_passes(arm):
    on, rows_on = _fit_counting(arm, True)
    off, rows_off = _fit_counting(arm, False)
    assert on.design_cached_ and not off.design_cached_
    assert np.array_equal(on.w_, off.w_) and on.ridge_ == off.ridge_ and on.val_mse_ == off.val_mse_
    n_train, n_val = 1399, 499                  # one row is lost to the one-step shift in each split
    assert rows_off == 2 * n_train + n_val      # old behaviour: train twice
    assert rows_on == n_train + n_val           # now: train once


def test_design_cache_is_skipped_when_it_would_not_fit_in_memory():
    from src.baselines.feature_map_forecaster import FeatureMapForecaster
    m = FeatureMapForecaster("quantum", 4, 2, "ring", 1.0, 0)
    m.CACHE_MAX_BYTES = 10
    assert m._train_design_chunks(np.zeros((100, 6))) is None and m.design_cached_ is False
