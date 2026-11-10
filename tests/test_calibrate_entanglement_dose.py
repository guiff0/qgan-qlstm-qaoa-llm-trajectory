import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.calibrate_entanglement_dose import config_block, gate_entropy, gate_positive_control, tag_runs


def _runs(entropy_by_dose, seeds=range(100, 105), pc_asr=70.0, anchor_asr=40.0, noise=0.01):
    r = np.random.default_rng(0); rows = []
    for k, e in enumerate(entropy_by_dose):
        for s in seeds:
            rows.append({"run_id": f"qgan_llm_dose_t{k}_rzz_20261110T010101Z", "seed": s,
                         "entanglement_entropy": e + r.normal(0, noise), "asr": anchor_asr})
    for s in seeds:
        rows.append({"run_id": "qgan_llm_dose_pc_untrained_20261110T010101Z", "seed": s, "entanglement_entropy": 0.0, "asr": pc_asr})
    return tag_runs(pd.DataFrame(rows))


def test_monotone_dose_passes_and_nonmonotone_fails():
    g = gate_entropy(_runs([0, .1, .2, .3, .4]))
    assert g["passed"] is True and all(g["criteria"].values()) and g["spearman_rho"] > 0.9
    bad = gate_entropy(_runs([0, .15, .41, .10, .3]))                         # the L4 > L6 pattern seen in the first sweep
    assert bad["passed"] is False and bad["criteria"]["mean_curve_strictly_increasing"] is False
    assert gate_entropy(_runs([0, .1, .2]))["passed"] is False


def test_noisy_draws_do_not_fail_a_rising_mean_curve_but_a_flat_one_fails():
    r = np.random.default_rng(3)
    rows = []
    for k, m in enumerate([0.0, 0.3, 0.55, 0.7, 0.8]):                        # rising in expectation, noisy per draw
        for s in range(100, 105):
            rows.append({"run_id": f"qgan_llm_dose_t{k}_rzz_20261110T010101Z", "seed": s,
                         "entanglement_entropy": m + (0 if k == 0 else r.normal(0, 0.12)), "asr": 40.0})
    g = gate_entropy(tag_runs(pd.DataFrame(rows)))
    assert g["passed"] is True and g["seeds_strictly_ordered_info_only"] < 5      # per-seed strictness is informational only
    flat = [{"run_id": f"qgan_llm_dose_t{k}_rzz_20261110T010101Z", "seed": s,
             "entanglement_entropy": (0 if k == 0 else 0.4) + r.normal(0, 0.1) * (k > 0), "asr": 40.0}
            for k in range(5) for s in range(100, 105)]
    assert gate_entropy(tag_runs(pd.DataFrame(flat)))["passed"] is False         # plateau after theta=0: mean curve not increasing


def test_positive_control_gate_reads_only_anchor_and_pc():
    d = _runs([0, .1, .2, .3, .4])
    assert gate_positive_control(d, 10.0)["passed"] is True
    assert gate_positive_control(_runs([0, .1, .2, .3, .4], pc_asr=42.0), 10.0)["passed"] is False


def test_config_block_has_five_doses_with_rzz_zero_anchor():
    b = config_block()
    assert b.count("dose-T") == 5 and 'entanglement: "rzz:0.0"' in b and "dose-PC untrained" in b


def test_positive_control_block_is_the_theta0_architecture_with_the_untrained_flag():
    b = config_block()
    pc = b.split('"QGAN-LLM (dose-PC untrained)"')[1]
    assert 'entanglement: "rzz:0.0"' in pc and "untrained: true" in pc


def test_untrained_mode_is_wired_into_the_qgan_training_path():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "baselines", "qgan_llm.py"), encoding="utf8").read()
    assert 'self.config.get("untrained", False)' in src and "n_epochs = 0 if untrained" in src and "untrained_control" in src
