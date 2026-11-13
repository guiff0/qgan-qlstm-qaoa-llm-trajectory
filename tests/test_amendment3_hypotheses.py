"""Amendment 3 (12 Nov 2026): five hypotheses H1-H5, entanglement hypotheses (old H2, H5, M2) dropped, duplicate sweep level removed.
Synthetic rows stand in for results/all_results.csv; nothing is trained."""
import numpy as np
import pandas as pd

import scripts.run_hypothesis_tests as hyp
from src.evaluation import entanglement_design as ed
from src.evaluation.study_status import PREREG_NAMES, preregistered_slugs, hypothesis_budget
from src.evaluation.verdicts import EXPECTED, EXPLORATORY_NAMES, NAMES, compute_verdicts
from src.utils.run_identity import model_slug

SEEDS = list(range(53, 63))


def _df(rmse_gap=0.0, asr_gap=0.0, noise_asr_gap=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cond, mt, noise in (("QGAN-LLM", "QGAN-LLM", 0.01), ("Classical GAN-LLM", "Classical GAN-LLM", np.nan),
                            ("QGAN-LLM (no noise)", "QGAN-LLM", 0.0)):
        for i, s in enumerate(SEEDS):
            q = cond.startswith("QGAN")
            rows.append({"run_id": f"{model_slug(cond)}_20261113T{i:06d}Z", "model_type": mt, "seed": s, "noise_strength": noise,
                         "fid": 100 + 5 * q + rng.normal(0, 1), "forecast_path": "generator", "persistence_rmse": 1e-3,
                         "rmse": 1e-3 + (rmse_gap if q else 0) + rng.normal(0, 2e-6),
                         "asr": 40 + (asr_gap if q else 0) + (noise_asr_gap if cond == "QGAN-LLM (no noise)" else 0) + rng.normal(0, 1),
                         "poisoning_rmse_degradation_pct": 3 + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_sweep_has_no_level_equal_to_the_primary_condition():
    # primary QGAN-LLM = 12 qubits, ring, depth 4 (config/default_config.yaml); no sweep level may reproduce it
    assert ("ring", 4) not in {(t, d) for _, t, d in ed.LEVELS}
    assert len({(t, d) for _, t, d in ed.LEVELS}) == len(ed.LEVELS)
    assert [lv for lv, _, _ in ed.LEVELS] == list(range(len(ed.LEVELS)))
    counts = [ed.cnots_per_forward(12, t, d) for _, t, d in ed.LEVELS]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)


def test_primary_config_and_sweep_names_do_not_collide():
    import yaml
    cfg = yaml.safe_load(open("config/default_config.yaml", newline="").read())
    q = cfg["qgan_llm"]
    primary = (q["n_qubits"], q["entanglement"], q["n_layers"], q["noise_strength"])
    for a in cfg["ablations"]:
        if a["name"].startswith("QGAN-LLM (ent-L"):
            assert (a["n_qubits"], a["entanglement"], a["n_layers"], q["noise_strength"]) != primary, a["name"]


def test_exactly_five_hypotheses_and_entanglement_ones_are_dropped():
    assert [v for v in NAMES.values()] == ["H1", "H2", "H3", "H4", "H5"]
    assert set(EXPECTED) >= set(NAMES.values())
    assert not any("entanglement" in k or "mediation" in k or "tomography" in k for k in NAMES)
    assert all(v.startswith("X") for v in EXPLORATORY_NAMES.values())


def test_default_study_is_four_conditions_without_the_sweep():
    assert len(PREREG_NAMES) == 4 and not any("ent-L" in n for n in PREREG_NAMES)
    assert len(preregistered_slugs()) == 4
    b = hypothesis_budget({s: 10 for s in preregistered_slugs()})
    assert list(b) == ["H1", "H2", "H3", "H4", "H5"] and all(v["remaining_for_full_design"] == 0 for v in b.values())


def test_h2_and_h3_are_the_rmse_and_asr_halves_of_the_old_h4():
    df = _df(rmse_gap=-5e-5, asr_gap=-6.0)
    old = hyp.test_h4_quantum_advantage(df)
    h2, h3 = hyp.test_h2_forecast_rmse(df), hyp.test_h3_attack_success(df)
    assert h2["metric"] == "rmse" and h3["metric"] == "asr"
    assert h2["p_value"] == old["rmse"]["p_value"] and h3["p_value"] == old["asr"]["p_value"]
    v = compute_verdicts({"H2_forecast_rmse": h2, "H3_attack_success_qgan_vs_classical": h3})
    assert v["H2_forecast_rmse"]["verdict"] == "SUPPORTED" and v["H3_attack_success_qgan_vs_classical"]["verdict"] == "SUPPORTED"


def test_h2_and_h3_are_judged_independently():
    df = _df(rmse_gap=-5e-5, asr_gap=0.0, seed=2)              # quantum better on RMSE only
    v = compute_verdicts({"H2_forecast_rmse": hyp.test_h2_forecast_rmse(df),
                          "H3_attack_success_qgan_vs_classical": hyp.test_h3_attack_success(df)})
    assert v["H2_forecast_rmse"]["verdict"] == "SUPPORTED"
    assert v["H3_attack_success_qgan_vs_classical"]["verdict"] == "NOT_SUPPORTED"


def test_h5_noise_attack_success_detects_the_planted_effect_and_a_null():
    pos = hyp.test_h5_noise_attack_success(_df(noise_asr_gap=8.0))
    assert pos["design"] == "matched_config_noise_only" and pos["n_per_group_used"] == {"no_noise": 10, "noisy": 10}
    assert compute_verdicts({"H5_noise_attack_success": pos})["H5_noise_attack_success"]["verdict"] == "SUPPORTED"
    null = hyp.test_h5_noise_attack_success(_df(noise_asr_gap=0.0, seed=4))
    assert compute_verdicts({"H5_noise_attack_success": null})["H5_noise_attack_success"]["verdict"] == "NOT_SUPPORTED"
    wrong = hyp.test_h5_noise_attack_success(_df(noise_asr_gap=-8.0))
    assert compute_verdicts({"H5_noise_attack_success": wrong})["H5_noise_attack_success"]["verdict"] == "CONTRADICTED"


def test_h5_is_untestable_without_the_matched_no_noise_runs():
    df = _df()
    df = df[~df["run_id"].str.contains("no_noise")]
    r = hyp.test_h5_noise_attack_success(df)
    assert r["status"] == "insufficient_data" and r["design"] == "matched_pair_missing"
    assert compute_verdicts({"H5_noise_attack_success": r})["H5_noise_attack_success"]["verdict"] == "UNTESTABLE"


def test_exploratory_entanglement_results_never_enter_the_holm_family():
    df = _df(rmse_gap=-5e-5, asr_gap=-6.0)
    out = {"H2_forecast_rmse": hyp.test_h2_forecast_rmse(df),
           "X1_entanglement_vs_asr": {"r": -0.9, "p_value": 1e-6, "df": 40, "ci_low": -0.95, "ci_high": -0.8, "unit": "run",
                                      "min_runs_per_level": 6}}
    v = compute_verdicts(out)
    assert "p_holm" in v["H2_forecast_rmse"] and "p_holm" not in v["X1_entanglement_vs_asr"]
