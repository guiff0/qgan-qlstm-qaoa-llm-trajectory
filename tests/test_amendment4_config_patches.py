"""
Non-regression tests for Amendment 4 config patches.

Ensures that the three config changes required for H3/H4 decidability
are present and correct in config/default_config.yaml.  These tests
must stay green for every future config edit — a failure here means
a regression to an undecidable study.

PATCHES REQUIRED (all applied 2026-11-16):
  1. classical_gan_llm.attack_samples  50  → 200  (H3 ASR SE ±3–4pp, was ±10–15pp)
  2. classical_gan_llm.synthetic_ratio  0.4 → 1.0  (H4 poisoning attack not diluted)
  3. qgan_llm.attack_samples           50  → 200  (same H3 fix for quantum arm)
  4. qgan_llm.synthetic_ratio          0.4 → 1.0  (same H4 fix for quantum arm)
  5. poisoning.shift_sd                0.5 → 1.0  (matches calibration run 2026-11-13)
  6. poisoning.calibrated          false  → true   (activates protocol v2)
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── 1. attack_samples ──────────────────────────────────────────────────────────

class TestAttackSamples:
    """
    H3 ASR standard error scales as sqrt(asr*(1-asr)/n).  With attack_samples=50
    and typical ASR≈40 %, SE≈±10–15 pp — too wide to decide the 5 pp gate.
    With attack_samples=200 SE drops to ±3–4 pp, making the gate decidable.
    Regression: must never go below 200.
    """

    def test_classical_gan_llm_attack_samples_is_200(self, cfg):
        v = cfg["classical_gan_llm"]["attack_samples"]
        assert v == 200, (
            f"classical_gan_llm.attack_samples={v}; needs 200 for H3 to be decidable "
            f"(SE ≈ ±{(0.4*0.6/v)**0.5*100:.1f} pp at ASR=40 %; gate is 5 pp)"
        )

    def test_qgan_llm_attack_samples_is_200(self, cfg):
        v = cfg["qgan_llm"]["attack_samples"]
        assert v == 200, (
            f"qgan_llm.attack_samples={v}; needs 200 for H3 to be decidable "
            f"(SE ≈ ±{(0.4*0.6/v)**0.5*100:.1f} pp at ASR=40 %; gate is 5 pp)"
        )

    def test_attack_samples_match_across_arms(self, cfg):
        """Both arms must use the same sample count or the H3 comparison is unbalanced."""
        c = cfg["classical_gan_llm"]["attack_samples"]
        q = cfg["qgan_llm"]["attack_samples"]
        assert c == q, (
            f"attack_samples mismatch: classical={c} vs quantum={q}; "
            "both arms must be identical for a balanced H3 comparison"
        )

    def test_no_ablation_overrides_attack_samples_below_200(self, cfg):
        """Ablation entries that set attack_samples below 200 would invalidate H3 for that arm."""
        for abl in cfg.get("ablations", []):
            n = abl.get("attack_samples")
            if n is not None:
                assert n >= 200, (
                    f"Ablation '{abl.get('name')}' sets attack_samples={n} < 200; "
                    "H3 would be undecidable for this condition"
                )
        for abl in cfg.get("classical_ablations", []):
            n = abl.get("attack_samples")
            if n is not None:
                assert n >= 200, (
                    f"Classical ablation '{abl.get('name')}' sets attack_samples={n} < 200"
                )


# ── 2. synthetic_ratio ─────────────────────────────────────────────────────────

class TestSyntheticRatio:
    """
    poisoning_resistance.py line 128 reads model.config["synthetic_ratio"] to
    decide what fraction of training features are replaced by GAN-generated rows.
    With synthetic_ratio=0.4 only 40 % of rows carry the attack → main-arm
    degradation was 0.07–2.37 % (gate is 5 pp).  With synthetic_ratio=1.0 all
    features are synthetic and the attack propagates fully.

    NOTE: synthetic_ratio in LLM fine-tuning (nvidia_finetune.py) uses explicit
    arguments, NOT this config key — changing this value does NOT affect LLM
    fine-tuning, only poisoning evaluation.
    """

    def test_classical_gan_llm_synthetic_ratio_is_1_0(self, cfg):
        v = cfg["classical_gan_llm"]["synthetic_ratio"]
        assert v == 1.0, (
            f"classical_gan_llm.synthetic_ratio={v}; must be 1.0 so the poisoning "
            "attack is not diluted in evaluate_poisoning_resistance() "
            "(poisoning_resistance.py line 128)"
        )

    def test_qgan_llm_synthetic_ratio_is_1_0(self, cfg):
        v = cfg["qgan_llm"]["synthetic_ratio"]
        assert v == 1.0, (
            f"qgan_llm.synthetic_ratio={v}; must be 1.0 so the H4 poisoning gate "
            "is decidable (main-arm degradation was 0.07–2.37 % with ratio=0.4)"
        )

    def test_synthetic_ratio_match_across_arms(self, cfg):
        c = cfg["classical_gan_llm"]["synthetic_ratio"]
        q = cfg["qgan_llm"]["synthetic_ratio"]
        assert c == q, (
            f"synthetic_ratio mismatch: classical={c} vs quantum={q}; "
            "both arms must use the same ratio for a valid H4 comparison"
        )

    def test_synthetic_ratio_is_a_valid_probability(self, cfg):
        for section in ("classical_gan_llm", "qgan_llm"):
            v = cfg[section]["synthetic_ratio"]
            assert 0.0 <= v <= 1.0, f"{section}.synthetic_ratio={v} is not in [0, 1]"


# ── 3. poisoning section ───────────────────────────────────────────────────────

class TestPoisoningSection:
    """
    shift_sd controls the label-inversion magnitude in poisoning_protocol.py.
    The calibration run (poisoning_calibration_20261113T234854Z.json) confirmed
    that chosen_shift_sd=1.0 is needed to reach the 10 % PC degradation target.
    calibrated=true activates protocol v2 (chronological split + ridge head).
    """

    def test_poisoning_shift_sd_is_1_0(self, cfg):
        v = cfg["poisoning"]["shift_sd"]
        assert v == 1.0, (
            f"poisoning.shift_sd={v}; must be 1.0 to match calibration run "
            "2026-11-13 (chosen_shift_sd=1.0 for 10 % PC degradation target)"
        )

    def test_poisoning_calibrated_is_true(self, cfg):
        v = cfg["poisoning"]["calibrated"]
        assert v is True, (
            f"poisoning.calibrated={v!r}; must be True to activate protocol v2 "
            "(chronological split + ridge head) in poisoning_resistance.py"
        )

    def test_poisoning_section_has_required_keys(self, cfg):
        p = cfg["poisoning"]
        required = {"shift_sd", "calibrated", "fraction"}
        missing = required - set(p)
        assert not missing, f"poisoning section missing keys: {missing}"

    def test_poisoning_fraction_is_sensible(self, cfg):
        """Fraction of real targets poisoned; 0.05 is the pre-registered value."""
        v = cfg["poisoning"]["fraction"]
        assert 0.0 < v <= 0.5, f"poisoning.fraction={v} is outside (0, 0.5]"


# ── 4. Cross-section consistency ───────────────────────────────────────────────

class TestCrossSectionConsistency:
    """Structural checks that prevent silent misconfiguration."""

    def test_config_loads_without_error(self):
        with open(CONFIG_PATH) as f:
            c = yaml.safe_load(f)
        assert c is not None and isinstance(c, dict)

    def test_both_gan_sections_present(self, cfg):
        for section in ("classical_gan_llm", "qgan_llm", "poisoning"):
            assert section in cfg, f"Missing top-level config section: {section}"

    def test_qgan_llm_no_noise_ablation_present(self, cfg):
        """QGAN-LLM (no noise) is required for H5; must be in ablations."""
        names = [a.get("name", "") for a in cfg.get("ablations", [])]
        assert any("no noise" in n.lower() for n in names), (
            "No 'no noise' ablation found in config.ablations; H5 is untestable"
        )

    def test_classical_matched_ablation_present(self, cfg):
        """Classical GAN-LLM (matched) is required for H2; must be in classical_ablations."""
        names = [a.get("name", "") for a in cfg.get("classical_ablations", [])]
        assert any("matched" in n.lower() for n in names), (
            "No 'matched' ablation in config.classical_ablations; H2 bracket is missing"
        )
