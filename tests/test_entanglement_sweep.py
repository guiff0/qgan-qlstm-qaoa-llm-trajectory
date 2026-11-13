"""Optional exploratory entanglement sweep (old H2/H5/M2, dropped from the confirmatory plan 12 Nov 2026; now X1-X3).
Config generation, level parsing, wiring, verdicts.
No training is run; synthetic results rows stand in for results/all_results.csv."""
import numpy as np
import pandas as pd
import yaml

import scripts.gen_entanglement_sweep as gen
import scripts.run_hypothesis_tests as hyp
import scripts.run_repeated_experiment as rre
from src.evaluation import entanglement_design as ed
from src.evaluation.verdicts import compute_verdicts
from src.utils.run_identity import config_of_run_id, model_slug


# ------------------------------------------------------------------ helpers
def _sweep_df(n_seeds=6, entropy_slope=0.5, asr_per_entropy=-3.0, seed=0, noise=0.4, acc_per_entropy=0.02):
    rng = np.random.default_rng(seed)
    rows = []
    for level, topo, depth in ed.LEVELS:
        for k in range(n_seeds):
            ent = entropy_slope * level + rng.normal(0, noise)
            rows.append({
                "run_id": f"{model_slug(ed.sweep_name(level, topo, depth))}_20261031T{level:02d}{k:02d}00Z",
                "model_type": "QGAN-LLM", "seed": 43 + k,
                "entanglement_entropy": ent,
                "asr": 40 + asr_per_entropy * ent + rng.normal(0, 1.0),
                "accuracy": 0.6 + acc_per_entropy * ent + rng.normal(0, 0.01),
                "asr_baseline": "last_close",
            })
    df = pd.DataFrame(rows)
    df["llm_threat_accuracy"] = df["accuracy"]          # H5 prefers the LLM detector's accuracy column
    return df


# ------------------------------------------------------------ design module
def test_names_round_trip_through_the_run_id_slug():
    for level, topo, depth in ed.LEVELS:
        slug = model_slug(ed.sweep_name(level, topo, depth))
        assert ed.parse_sweep_config(slug + "_20261031T120000Z") == {"level": level, "topology": topo, "depth": depth}
        assert config_of_run_id(slug + "_20261031T120000Z") == slug


def test_non_sweep_configs_are_not_parsed_as_sweep():
    for rid in ("qgan_llm_20261031T120000Z", "qgan_llm_no_noise_20261031T120000Z",
                "qgan_llm_12_qubit_20261031T120000Z", "classical_gan_llm_20261031T120000Z"):
        assert ed.parse_sweep_config(rid) is None


def test_level_zero_is_a_true_zero_entanglement_control():
    lv0 = [lv for lv in ed.LEVELS if lv[0] == 0]
    assert lv0 == [(0, "none", 4)]
    assert ed.cnots_per_forward(12, "none", 4) == 0
    # same depth (so same parameter count) as the first ring level: only the entangling gates differ
    assert {d for _, t, d in ed.LEVELS if t in ("none", "ring")} == {4}


def test_level_order_is_monotone_in_entangling_gates():
    counts = [ed.cnots_per_forward(12, t, d) for _, t, d in ed.LEVELS]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)


# --------------------------------------------------------------- generator
def test_generated_block_is_valid_yaml_with_fixed_qubits_and_idempotent(tmp_path):
    src = open("config/default_config.yaml", newline="").read()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(src, newline="")
    once = gen.apply_block(src, gen.build_block(12))
    twice = gen.apply_block(once, gen.build_block(12))
    assert once == twice                                    # idempotent
    parsed = yaml.safe_load(once)
    sweep = [a for a in parsed["ablations"] if a["name"].startswith("QGAN-LLM (ent-L")]
    assert [a["name"] for a in sweep] == ed.all_sweep_names()
    assert {a["n_qubits"] for a in sweep} == {12}            # width held fixed
    assert all("noise_strength" not in a for a in sweep)     # noise inherited, not varied
    is_sweep = lambda a: a["name"].startswith("QGAN-LLM (ent-L")           # noqa: E731
    # the original (non-sweep) ablations are untouched and keep their order, whether or not src already had the block
    assert [a["name"] for a in parsed["ablations"] if not is_sweep(a)] == \
           [a["name"] for a in yaml.safe_load(src)["ablations"] if not is_sweep(a)]
    assert parsed["qlstm_forecaster"]["n_qubits"] == 12      # rest of the file untouched (primary width, 12 since 12 Nov)
    # regenerating for another width replaces, not appends
    other = gen.apply_block(once, gen.build_block(8))
    assert sum(1 for a in yaml.safe_load(other)["ablations"] if a["name"].startswith("QGAN-LLM (ent-L")) == len(ed.LEVELS)


# ----------------------------------------------------------- harness match
def test_harness_matches_ablation_rows_by_run_id_not_model_type(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "all_results.csv").write_text(
        "run_id,model_type,seed\n"
        "qgan_llm_20261031T000001Z,QGAN-LLM,43\n"
        "qgan_llm_ent_l3_linear_d4_20261031T000002Z,QGAN-LLM,44\n")
    assert rre._csv_seeds("QGAN-LLM") == {43}
    assert rre._csv_seeds("QGAN-LLM (ent-L3 linear-d4)") == {44}
    assert rre._csv_seeds("QGAN-LLM (ent-L4 ring-d4)") == set()


# --------------------------------------------------------------- H2 / H5 / M2
def test_h2_uses_the_controlled_sweep_and_finds_the_planted_effect():
    r = hyp.test_h2_adversarial_resilience(_sweep_df())
    assert r["design"] == "controlled_sweep" and r["unit"] == "run"
    assert r["df"] + 2 == 6 * len(ed.LEVELS) and r["r"] < -0.9 and r["p_value"] < 1e-6
    assert r["min_runs_per_level"] == 6 and r["n_levels"] == len(ed.LEVELS)
    assert r["manipulation_check_level_vs_entropy"]["spearman_rho"] > 0.8
    assert r["dose_response_level_vs_outcome"]["spearman_rho"] < -0.8


def test_h2_ignores_non_sweep_rows_when_a_sweep_exists():
    df = pd.concat([_sweep_df(), pd.DataFrame([
        {"run_id": "qgan_llm_20261031T999999Z", "model_type": "QGAN-LLM", "seed": 42,
         "entanglement_entropy": 99.0, "asr": 99.0, "accuracy": 0.1, "asr_baseline": "last_close"}])],
        ignore_index=True)
    assert hyp.test_h2_adversarial_resilience(df)["df"] + 2 == 6 * len(ed.LEVELS)


def test_h2_falls_back_to_pooled_configs_when_the_sweep_is_too_small():
    small = _sweep_df(n_seeds=1)                               # 6 runs < 12
    r = hyp.test_h2_adversarial_resilience(small)
    assert r["design"] == "pooled_configs" and r["unit"] == "config"


def test_h5_reads_the_sweep_too():
    r = hyp.test_h5_tomography_correlation(_sweep_df())
    assert r["design"] == "controlled_sweep" and r["r"] > 0.9


def test_m2_mediation_on_the_sweep_recovers_the_planted_path():
    r = hyp.test_m2_dv2_mediation(_sweep_df())
    assert r["unit"] == "run" and r["n_observations"] == 6 * len(ed.LEVELS)
    assert r["path_a"] > 0.8 and r["path_b"] < -0.5 and r["ci_high"] < 0       # level -> entropy up, entropy -> ASR down
    assert r["standardized"] is True


# ---------------------------------------------------------------- verdicts
def _verdicts(df):
    out = {"X1_entanglement_vs_asr": hyp.test_h2_adversarial_resilience(df),
           "X2_entanglement_vs_detection": hyp.test_h5_tomography_correlation(df),
           "X3_entanglement_mediation": hyp.test_m2_dv2_mediation(df)}
    return compute_verdicts(out)


def test_sweep_with_enough_runs_is_confirmatory_and_decisive():
    v = _verdicts(_sweep_df())
    assert v["X1_entanglement_vs_asr"]["tier"] == "CONFIRMATORY"
    assert v["X1_entanglement_vs_asr"]["verdict"] == "SUPPORTED"          # r < 0 as hypothesised
    assert v["X2_entanglement_vs_detection"]["verdict"] == "SUPPORTED"
    assert v["X3_entanglement_mediation"]["tier"] == "CONFIRMATORY" and v["X3_entanglement_mediation"]["verdict"] == "SUPPORTED"
    assert not any("model CONFIG" in b for d in v.values() for b in d["blockers"])


def test_h5_on_the_interim_classical_detector_stays_exploratory():
    df = _sweep_df().drop(columns=["llm_threat_accuracy"])
    d = _verdicts(df)["X2_entanglement_vs_detection"]
    assert d["tier"] == "EXPLORATORY" and d["verdict"] == "SUGGESTIVE_FOR"
    assert any("interim classical" in b for b in d["blockers"])


def test_a_real_null_becomes_not_supported_not_inconclusive():
    v = _verdicts(_sweep_df(asr_per_entropy=0.0, acc_per_entropy=0.0, seed=3))
    assert v["X1_entanglement_vs_asr"]["tier"] == "CONFIRMATORY"
    assert v["X1_entanglement_vs_asr"]["verdict"] == "NOT_SUPPORTED"


def test_thin_levels_keep_the_verdict_exploratory():
    v = _verdicts(_sweep_df(n_seeds=2))                                        # 12 runs, 2 per level
    d = v["X1_entanglement_vs_asr"]
    assert d["tier"] == "EXPLORATORY" and any("thinnest sweep level" in b for b in d["blockers"])


def test_failed_manipulation_check_blocks_a_confirmatory_verdict():
    df = _sweep_df(entropy_slope=0.0, noise=1.0, seed=5)        # level does not move the measured entropy
    d = _verdicts(df)["X1_entanglement_vs_asr"]
    assert d["tier"] == "EXPLORATORY" and any("manipulation check failed" in b for b in d["blockers"])


def test_legacy_pooled_rows_still_get_the_config_blocker():
    rows = []
    for i, name in enumerate(["qgan_llm", "qgan_llm_12_qubit", "qgan_llm_linear_topology", "qgan_llm_no_noise", "qlstm"]):
        rows.append({"run_id": f"{name}_20261031T00000{i}Z", "model_type": "QGAN-LLM", "seed": 42,
                     "entanglement_entropy": float(i), "asr": 10.0 - i + 0.1 * (i % 2), "accuracy": 0.5 + 0.01 * i,
                     "asr_baseline": "last_close"})
    d = _verdicts(pd.DataFrame(rows))["X1_entanglement_vs_asr"]
    assert d["tier"] == "EXPLORATORY" and any("one model config" in b for b in d["blockers"])
