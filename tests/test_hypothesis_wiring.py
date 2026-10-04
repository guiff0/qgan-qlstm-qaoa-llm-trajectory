"""
Regression tests for the H3 (poisoning/model-inversion), H5 (threat
detection/FPR), and quantum resilience-suite wiring -- these previously
existed as unwired modules (attacks/poisoning.py, attacks/threat_labels.py,
attacks/quantum_attacks.py, quantum/gradient_obfuscation.py,
quantum/encoding_fidelity.py, metrics/resilience.py) called only from
their own unit tests, never from the actual training/evaluation pipeline.
"""
import numpy as np
import torch
import torch.nn as nn

from src.evaluation.poisoning_resistance import evaluate_poisoning_resistance
from src.evaluation.threat_detection import evaluate_threat_detection
from src.evaluation.resilience_suite import run_resilience_suite


class _FakeGenerativeModel:
    """Minimal stand-in for QGANLLM/ClassicalGANLLM's generator interface."""
    def __init__(self, n_features, synthetic_ratio=0.3):
        self.config = {"synthetic_ratio": synthetic_ratio}

    def generate_synthetic_data(self, n):
        return np.random.randn(n, 8).astype(np.float32)


class _FakeNonGenerativeModel:
    """Minimal stand-in for QLSTMForecaster -- no generator at all."""
    def __init__(self):
        self.config = {}


def _toy_data(n=300, f=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float32)
    y = (X[:, 0] + 0.05 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def test_poisoning_resistance_with_generator():
    X, y = _toy_data()
    model = _FakeGenerativeModel(n_features=8)
    result = evaluate_poisoning_resistance(model, X, y, clean_rmse=0.2, seed=1)
    assert np.isfinite(result["poisoned_rmse"])
    assert np.isfinite(result["poisoning_rmse_degradation_pct"])
    assert result["poisoning_n_poisoned"] > 0
    assert 0.0 <= result["model_inversion_accuracy"] <= 1.0


def test_poisoning_resistance_without_generator_degrades_gracefully():
    X, y = _toy_data()
    model = _FakeNonGenerativeModel()
    result = evaluate_poisoning_resistance(model, X, y, clean_rmse=0.2, seed=1)
    assert np.isfinite(result["poisoned_rmse"])
    assert result["poisoning_synthetic_ratio_used"] == 0.0
    assert np.isnan(result["model_inversion_accuracy"])  # honestly N/A, not a fabricated number


def test_threat_detection_produces_fpr_in_valid_range():
    X, y = _toy_data(n=400)
    head = nn.Linear(8, 1)
    head.eval()
    attack_cfg = {"fgsm_epsilon": 0.1, "pgd_epsilon": 0.1, "pgd_alpha": 0.01, "pgd_steps": 5}
    result = evaluate_threat_detection(head, X, y, attack_cfg, seed=1,
                                        n_clean=80, n_fgsm=40, n_pgd=40)
    assert 0.0 <= result["fpr"] <= 1.0
    assert result["threshold"] == 0.7  # the dissertation's QACL threshold, not classification_metrics' 0.5 default
    assert result["detector"] == "interim_classical_randomforest"


def test_resilience_suite_returns_finite_values_including_ctr_at_small_qubit_count():
    n_qubits, n_layers = 4, 2
    n_params = n_qubits * n_layers * 3
    weights = np.random.randn(n_params) * 0.1
    X_single = np.random.randn(8)
    X_batch = np.random.randn(20, 8)
    head = nn.Linear(8, 1)
    head.eval()

    result = run_resilience_suite(head, weights, X_single, 0.1, X_batch,
                                   n_qubits, n_layers, "ring",
                                   dev_name="default.qubit", seed=0)
    for key in ["qsfr", "eer", "nlcs", "qgom", "m1_efi", "ctr"]:
        assert np.isfinite(result[key]), f"{key} should be a finite float, got {result[key]}"


def test_resilience_suite_skips_ctr_gracefully_above_12_qubits():
    n_qubits, n_layers = 14, 1
    weights = np.random.randn(n_qubits * n_layers * 3) * 0.1
    X_single = np.random.randn(16)
    X_batch = np.random.randn(5, 16)
    head = nn.Linear(16, 1)
    head.eval()

    result = run_resilience_suite(head, weights, X_single, 0.1, X_batch,
                                   n_qubits, n_layers, "ring",
                                   dev_name="default.qubit", seed=0)
    assert np.isnan(result["ctr"])
    assert "exceeds" in result["ctr_note"]
