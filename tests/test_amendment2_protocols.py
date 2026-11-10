"""Amendment 2: H3 positive control, H5 detector instrument, equivalence power note, exploratory mediator screen.

Everything here is numpy/scipy only (no torch, pennylane or network). The LLM is a stub that reads the numbers in the prompt.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attacks.poisoning import inject_triggered_shift, invert_continuous_targets  # noqa: E402
from src.evaluation import poisoning_protocol as pp  # noqa: E402
from src.evaluation.decidability import detector_gate, poisoning_gate  # noqa: E402
from src.evaluation.equivalence_power import runs_needed_for_equivalence  # noqa: E402
from src.evaluation.mediation_screen import holm, screen  # noqa: E402
from src.llm import llm_pipeline as lp  # noqa: E402
from src.llm import threat_calibration as tc  # noqa: E402


def _data(n=20000, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ (rng.normal(size=d) * 0.02) + rng.normal(size=n)
    return X[: n // 2], y[: n // 2], X[n // 2:], y[n // 2:]


# ------------------------------------------------------------------ H3 ----
def test_legacy_reflection_is_a_noop_and_v2_is_not():
    Xt, yt, Xe, ye = _data()
    a = invert_continuous_targets(yt, fraction=0.05, seed=1)
    c = pp.rmse(pp.predict(pp.fit_ridge(Xt, yt), Xe), ye)
    p = pp.rmse(pp.predict(pp.fit_ridge(Xt, a["y_poisoned"]), Xe), ye)
    assert abs((p - c) / c * 100) < 0.2                      # the old attack: indistinguishable from nothing
    d = pp.degradation(Xt, yt, Xe, ye, fraction=0.2, shift_sd=3.0, seed=1)["degradation_pct"]
    assert d > 5.0                                           # the new one clears the H3 margin on the control head


def test_protocol_is_deterministic_and_the_positive_control_uses_no_generator():
    Xt, yt, Xe, ye = _data()
    cfg = {"fraction": 0.2, "shift_sd": 3.0}
    a = pp.run_protocol(Xt, yt, Xe, ye, X_real_train=Xt, seed=5, cfg=cfg)
    b = pp.run_protocol(Xt, yt, Xe, ye, X_real_train=Xt, seed=5, cfg=cfg)
    assert a == b
    assert a["poisoning_protocol"] == pp.PROTOCOL_NAME and "poisoning_pc_degradation_pct" in a
    # the control is computed from the real rows, so replacing the mixed rows must not change it
    noisy = Xt + np.random.default_rng(3).normal(size=Xt.shape)
    c = pp.run_protocol(noisy, yt, Xe, ye, X_real_train=Xt, seed=5, cfg=cfg)
    assert c["poisoning_pc_degradation_pct"] == a["poisoning_pc_degradation_pct"]
    assert c["poisoning_rmse_degradation_pct"] != a["poisoning_rmse_degradation_pct"]


def test_extra_feature_variance_reduces_damage_so_h3_has_something_to_detect():
    Xt, yt, Xe, ye = _data()
    noisy = Xt + np.random.default_rng(3).normal(size=Xt.shape)
    clean = np.mean([pp.degradation(Xt, yt, Xe, ye, fraction=0.2, shift_sd=3, seed=s)["degradation_pct"] for s in range(6)])
    nz = np.mean([pp.degradation(noisy, yt, Xe, ye, fraction=0.2, shift_sd=3, seed=s)["degradation_pct"] for s in range(6)])
    assert nz < clean


def test_calibration_returns_the_smallest_shift_reaching_the_target_and_none_when_unreachable():
    Xt, yt, Xe, ye = _data()
    out = pp.calibrate_shift(Xt, yt, Xe, ye, seeds=[100, 101, 102], target_pct=10.0)
    assert out["chosen_shift_sd"] is not None and out["pilot_seeds"] == [100, 101, 102]
    meds = {r["shift_sd"]: r["median_degradation_pct"] for r in out["curve"]}
    assert meds[out["chosen_shift_sd"]] >= 10.0 and all(m < 10.0 for s, m in meds.items() if s < out["chosen_shift_sd"])
    none = pp.calibrate_shift(Xt, yt, Xe, ye, seeds=[100], grid=(0.1,), target_pct=1000.0)
    assert none["chosen_shift_sd"] is None and "not decidable" in none["note"]


def test_triggered_shift_validates_and_leaves_features_alone():
    Xt, yt, _, _ = _data(2000)
    r = inject_triggered_shift(Xt, yt, fraction=0.1, shift_sd=2.0, seed=0)
    assert r["n_poisoned"] == int(len(yt) * 0.1) and (r["y_poisoned"] != yt).sum() == r["n_poisoned"]
    with pytest.raises(ValueError):
        inject_triggered_shift(Xt, yt, fraction=1.5)


def test_poisoning_gate_positive_control():
    moving = [2.0, 3.0, -1.0]                                    # the outcome moves a little
    g = poisoning_gate(moving, 5.0, positive_control=[0.01, 0.02, 0.0])
    assert g["decidable"] is False and g["gate"] == "poisoning_positive_control"
    ok = poisoning_gate(moving, 5.0, positive_control=[9.0, 11.0, 12.0])
    assert ok["decidable"] is True
    assert poisoning_gate(moving, 5.0)["decidable"] is True      # legacy call unchanged
    assert poisoning_gate([0.0003, -0.0002], 5.0)["decidable"] is False


# ------------------------------------------------------------------ H5 ----
def test_auroc_and_threshold():
    assert tc.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert tc.auroc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(tc.auroc([1, 1], [0.1, 0.2]))
    t = tc.pick_threshold([0, 0, 1, 1], [0.1, 0.2, 0.4, 0.5])
    assert t["calibrated"] and 0.2 < t["threshold"] < 0.4
    flat = tc.pick_threshold([0, 1, 0, 1], [0.8, 0.8, 0.8, 0.8])
    assert not flat["calibrated"] and flat["threshold"] == 0.70


def test_split_is_disjoint_and_complete():
    s = tc.split_windows(102, 0.3, 4, seed=3)
    parts = [set(s[k].tolist()) for k in ("shot", "cal", "eval")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert sum(len(p) for p in parts) == 102 and len(parts[0]) == 4


def test_detector_gate_positive_control_blocks_a_detector_that_misses_gross_perturbations():
    acc = np.full(10, 0.62); n = np.full(10, 200)                # above chance, so the old gate passes
    assert detector_gate(acc, n)["decidable"] is True
    blocked = detector_gate(acc, n, pc_passed=[False] * 9 + [True])
    assert blocked["decidable"] is False and blocked["gate"] == "detector_positive_control"
    assert detector_gate(acc, n, pc_passed=[True] * 10)["decidable"] is True
    assert detector_gate(np.full(10, 0.501), n)["decidable"] is False


def _series(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    return 1.10 + np.cumsum(rng.normal(0, 5e-5, n))


def _cfg(**kw):
    base = {"model_id": "m", "n_windows": 60, "window_size": 10, "n_regimes": 3, "max_workers": 1}
    base.update(kw)
    return base


def _numbers(msgs):
    user = msgs[-1]["content"]
    return [float(x) for x in re.findall(r"\d\.\d{5}", user.split("A generative")[0].split("Predict")[0])]


def _informative(msgs):
    """Stub LLM: flags an abnormal final step, as a good detector would."""
    w = np.array(_numbers(msgs)); d = np.diff(w)
    z = abs(d[-1]) / (np.std(d[:-1]) + 1e-9)
    return f"Forecast: {w[-1]:.5f}\nThreat probability: {0.9 if z > 3.0 else 0.1}"


def _uninformative(msgs):
    w = _numbers(msgs)
    return f"Forecast: {w[-1]:.5f}\nThreat probability: 0.8"


def _run(chat, **kw):
    c = _series()
    return lp.evaluate_llm_pipeline(c, np.r_[c[1:], c[-1]], None, _cfg(**kw), seed=3, chat_fn=chat, cache_dir=None)


def test_legacy_columns_are_unchanged_when_the_instrument_is_off():
    r = _run(_informative)
    assert not any(k in r for k in ("llm_threat_accuracy_cal", "llm_threat_pc_passed", "ref_last_jump_auroc"))
    assert r["llm_n_calls"] == 2 * r["llm_n_windows"]


def test_instrument_separates_a_real_detector_from_a_chance_detector():
    flags = dict(threat_calibration=True, threat_cal_fraction=0.3, threat_few_shot=4, threat_positive_control=True,
                 threat_pc_k=20.0, threat_pc_windows=20)
    good, bad = _run(_informative, **flags), _run(_uninformative, **flags)
    assert good["llm_threat_pc_passed"] is True and good["llm_threat_pc_accuracy"] >= 0.75
    assert bad["llm_threat_pc_passed"] is False and abs(bad["llm_threat_auroc"] - 0.5) < 1e-9
    assert bad["llm_threat_accuracy_cal"] <= 0.55            # a score that never varies cannot beat chance
    assert good["ref_last_jump_auroc"] > 0.8                 # the signal exists in the window itself
    # shots and pc prompts are extra calls, but never counted as scored windows
    assert good["llm_threat_few_shot"] == 8 and good["llm_threat_cal_n"] > 0
    assert good["llm_n_calls"] == 2 * good["llm_n_windows"] + 20


def test_few_shot_windows_are_not_scored_and_the_prompt_carries_the_examples():
    seen = []

    def spy(msgs):
        seen.append(msgs[0]["content"]); return _informative(msgs)

    r = _run(spy, threat_few_shot=4)
    assert all("Labelled reference windows" in s for s in seen)
    assert r["llm_n_forecasts"] <= r["llm_n_windows"] - 4 and r["llm_threat_n"] <= 2 * (r["llm_n_windows"] - 4)


# ------------------------------------------------------------------ H2 / M2 ----
def test_equivalence_power_note_matches_the_hand_calculation():
    assert runs_needed_for_equivalence(0.0) == 32
    assert runs_needed_for_equivalence(-0.085) == 57
    assert runs_needed_for_equivalence(-0.33) is None


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert np.all(adj >= np.array([0.01, 0.04, 0.03, 0.5])) and np.all(adj <= 1.0)


def test_screen_finds_a_planted_mediator_and_rejects_noise():
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(10):
        for level in (0, 2, 4, 6):
            m = 0.6 * level + rng.normal(0, 1)
            rows.append({"seed": seed, "sweep_level": level, "asr": 2.0 * m + rng.normal(0, 1),
                         "planted": m, "noise_a": rng.normal(), "noise_b": rng.normal()})
    res = screen(pd.DataFrame(rows), candidates=["planted", "noise_a", "noise_b"], n_bootstrap=400)
    assert res["survivors"] == ["planted"] and "EXPLORATORY" in res["label"]
    assert res["k_candidates_tested"] == 3
    assert set(res["discovery_seeds"]).isdisjoint(res["replication_seeds"])
