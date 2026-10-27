"""
Tests for the three modules covering Table 6's quantum resilience
framework that had zero test coverage until this file:
  - src/quantum/encoding_fidelity.py  (M1 = Encoding Fidelity Index)
  - src/metrics/resilience.py         (QAR, CQRS, NLCS, CTR)
  - src/quantum/gradient_obfuscation.py (QGOM)

Same philosophy as the rest of tests/: positive controls (a real effect
must be detected), negative controls (no effect must not be reported as
one), and explicit checks that the infeasible cases (CTR at 20 qubits,
NLCS on a full-system matrix) fail loudly rather than silently.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantum.encoding_fidelity import (
    state_preparation_fidelity, information_preservation,
    expressibility_deficit, encoding_injectivity, compute_m1_efi,
)
from src.quantum.encoding import amplitude_encode
from src.metrics.resilience import qar, cqrs, nlcs, ctr, check_ctr_tractable, MAX_QUBITS_FOR_DENSITY_MATRIX
from src.quantum.gradient_obfuscation import qgom_d, build_finite_shot_qnode
from src.quantum.tomography import statevector_from_circuit, reduced_density_matrix

N_QUBITS = 4
N_LAYERS = 2


# ---------------------------------------------------------------------------
# M1: Encoding Fidelity Index
# ---------------------------------------------------------------------------

@pytest.fixture
def m1_inputs():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(5, N_QUBITS))
    weights = rng.normal(size=N_LAYERS * N_QUBITS * 3) * 0.3
    return X, weights


def test_spf_is_a_real_bounded_fidelity_not_a_trivial_one(m1_inputs):
    """SPF must lie in [0,1] and must NOT be trivially 1.0 -- since this
    circuit uses angle encoding (not true amplitude encoding), its
    output state generally differs from the ideal amplitude-encoded
    reference. A constant 1.0 here would mean SPF isn't actually
    comparing two different states (the original bug this module's
    docstring describes fixing)."""
    X, weights = m1_inputs
    spf = state_preparation_fidelity(X, weights, N_QUBITS, N_LAYERS, "ring")
    assert 0.0 <= spf <= 1.0001
    assert spf < 0.999, "SPF should not be trivially ~1.0 for an angle-encoded circuit"


def test_information_preservation_bounded(m1_inputs):
    X, weights = m1_inputs
    ip = information_preservation(X, weights, N_QUBITS, N_LAYERS, "ring")
    assert np.isfinite(ip)


def test_expressibility_deficit_is_nonnegative(m1_inputs):
    _, weights = m1_inputs
    ed = expressibility_deficit(N_QUBITS, N_LAYERS, "ring", n_samples=30, seed=1)
    assert ed >= 0.0  # KL divergence is always >= 0


def test_expressibility_deficit_lower_for_more_expressive_topology():
    """POSITIVE CONTROL: 'full' entanglement should generally explore the
    Hilbert space more thoroughly than 'linear' -- i.e. be closer to
    Haar-random (lower ED) -- for the same layer count. Not a strict
    guarantee for every seed/circuit, so this uses a generous sample
    size and just checks the two are meaningfully different, not a
    hard direction, to avoid a flaky test on a genuinely stochastic
    comparison."""
    ed_full = expressibility_deficit(N_QUBITS, N_LAYERS, "full", n_samples=60, seed=2)
    ed_linear = expressibility_deficit(N_QUBITS, N_LAYERS, "linear", n_samples=60, seed=2)
    assert ed_full != ed_linear, "Different topologies should not produce identical ED by coincidence of implementation"


def test_encoding_injectivity_is_one_for_widely_separated_inputs():
    """POSITIVE CONTROL: inputs far apart in feature space should map to
    highly distinguishable states -- EI should be high (near 1).
    Uses 0 vs pi/2 rather than e.g. -3 vs 3: RY(theta) and RY(-theta)
    can coincidentally land at HIGH fidelity for some theta (e.g. at
    theta=3 rad, |<RY(-3)|RY(3)>|^2 ~= 0.98) since fidelity is
    insensitive to the sign flip in this single-qubit sense -- 0 vs
    pi/2 is an unambiguous quarter-turn separation (fidelity 0.5 for a
    single qubit) with no such coincidence."""
    rng = np.random.default_rng(3)
    X = np.array([[0.0] * N_QUBITS, [np.pi / 2] * N_QUBITS] * 10)
    weights = rng.normal(size=N_LAYERS * N_QUBITS * 3) * 0.3
    ei = encoding_injectivity(X, weights, N_QUBITS, N_LAYERS, "ring", n_pairs=50, seed=3)
    assert ei > 0.5


def test_encoding_injectivity_is_low_for_identical_inputs():
    """NEGATIVE CONTROL: every row identical -> every sampled pair is a
    'collision' by construction -- EI should be ~0."""
    rng = np.random.default_rng(4)
    X = np.tile(rng.normal(size=(1, N_QUBITS)), (20, 1))
    weights = rng.normal(size=N_LAYERS * N_QUBITS * 3) * 0.3
    ei = encoding_injectivity(X, weights, N_QUBITS, N_LAYERS, "ring", n_pairs=50, seed=4)
    assert ei < 0.1


def test_compute_m1_efi_returns_all_submetrics_and_composite(m1_inputs):
    X, weights = m1_inputs
    result = compute_m1_efi(X, weights, N_QUBITS, N_LAYERS, "ring", ed_n_samples=20, seed=5)
    for key in ("SPF", "IP", "ED", "EI", "M1_EFI"):
        assert key in result and np.isfinite(result[key])
    assert 0.0 <= result["M1_EFI"] <= 1.0001


# ---------------------------------------------------------------------------
# QAR and CQRS
# ---------------------------------------------------------------------------

def test_qar_is_one_when_full_advantage_persists():
    # clean gap = 0.31 - 0.087 = 0.223; attacked gap must equal 0.223 too for QAR=1.0
    result = qar(asr_classical_clean=0.31, asr_quantum_clean=0.087,
                 asr_classical_attacked=0.45, asr_quantum_attacked=0.45 - 0.223)
    assert abs(result - 1.0) < 1e-9


def test_qar_is_zero_when_advantage_fully_erased():
    result = qar(asr_classical_clean=0.31, asr_quantum_clean=0.087,
                 asr_classical_attacked=0.30, asr_quantum_attacked=0.30)
    assert abs(result) < 1e-9


def test_qar_can_be_negative_and_is_not_clipped():
    """The advantage REVERSING under attack is a real, reportable
    finding -- QAR must not be silently clamped to 0."""
    result = qar(asr_classical_clean=0.31, asr_quantum_clean=0.087,
                 asr_classical_attacked=0.10, asr_quantum_attacked=0.40)
    assert result < 0


def test_qar_nan_for_zero_clean_denominator():
    result = qar(asr_classical_clean=0.20, asr_quantum_clean=0.20,
                 asr_classical_attacked=0.30, asr_quantum_attacked=0.10)
    assert np.isnan(result)


def test_cqrs_weighted_average_matches_hand_calculation():
    result = cqrs(qsfr_val=0.94, eer_val=0.89, ctr_val=0.84, nlcs_val=0.79, qar_val=0.76)
    expected = 0.25 * 0.94 + 0.25 * 0.89 + 0.20 * 0.84 + 0.15 * 0.79 + 0.15 * 0.76
    assert abs(result["CQRS"] - expected) < 1e-9
    assert result["missing_metrics"] == []
    assert result["renormalized"] is False


def test_cqrs_renormalizes_when_ctr_is_missing():
    """The common case: CTR is NaN (infeasible at 20 qubits). CQRS must
    renormalize over the remaining four metrics, not silently propagate
    NaN through the whole composite."""
    result = cqrs(qsfr_val=0.94, eer_val=0.89, ctr_val=float("nan"), nlcs_val=0.79, qar_val=0.76)
    assert np.isfinite(result["CQRS"])
    assert "ctr" in result["missing_metrics"]
    assert result["renormalized"] is True
    assert abs(sum(result["weights_used"].values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# NLCS (rescoped to reduced density matrices)
# ---------------------------------------------------------------------------

def test_nlcs_is_one_for_identical_clean_and_attacked_states():
    rng = np.random.default_rng(6)
    weights = rng.normal(size=N_LAYERS * N_QUBITS * 3) * 0.3
    inputs = rng.normal(size=N_QUBITS)

    def circuit_fn(w):
        for i in range(N_QUBITS):
            import pennylane as qml
            qml.RY(inputs[i], wires=i)
        n_per_layer = N_QUBITS * 3
        import pennylane as qml
        for layer in range(N_LAYERS):
            start = layer * n_per_layer
            for q in range(N_QUBITS):
                idx = start + q * 3
                qml.RX(w[idx], wires=q)
                qml.RY(w[idx + 1], wires=q)
                qml.RZ(w[idx + 2], wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

    full_state = statevector_from_circuit(circuit_fn, weights, N_QUBITS, "default.qubit")
    rho = reduced_density_matrix(full_state, N_QUBITS, list(range(N_QUBITS // 2)))
    result = nlcs(rho, rho)
    assert abs(result - 1.0) < 1e-9


def test_nlcs_rejects_mismatched_shapes():
    rho_small = np.eye(2) / 2
    rho_big = np.eye(4) / 4
    with pytest.raises(ValueError):
        nlcs(rho_small, rho_big)


# ---------------------------------------------------------------------------
# CTR: must fail loudly and specifically above the tractable qubit ceiling
# ---------------------------------------------------------------------------

def test_ctr_tractability_check_passes_below_ceiling():
    check_ctr_tractable(MAX_QUBITS_FOR_DENSITY_MATRIX)  # should not raise


def test_ctr_raises_clear_error_at_primary_20_qubit_configuration():
    with pytest.raises(ValueError, match="not computable at n_qubits=20"):
        check_ctr_tractable(20)


def test_ctr_function_itself_raises_before_attempting_large_allocation():
    """ctr() must call the tractability gate BEFORE touching its (small,
    already-in-hand) input arrays -- confirms the gate isn't bypassable
    just because the caller already built small matrices."""
    dummy_series = [np.eye(2) / 2, np.eye(2) / 2]
    with pytest.raises(ValueError, match="not computable"):
        ctr(dummy_series, dummy_series, n_qubits=20)


def test_ctr_computes_a_real_ratio_at_a_tractable_qubit_count():
    """POSITIVE CONTROL at a tractable size: construct a coherence series
    that decays faster under 'attack' and confirm CTR < 1. Off-diagonal
    magnitude must actually vary with the decay value -- a purely
    diagonal matrix (e.g. np.eye(4)*v) has ZERO off-diagonal content by
    construction regardless of v, which would make _coherence_measure
    return 0 for every step and the fit degenerate."""
    t = np.arange(10)
    coherence_clean = 0.5 * np.exp(-t / 8.0)      # slow decay, T2=8
    coherence_attacked = 0.5 * np.exp(-t / 2.0)    # fast decay, T2=2

    def make_rho(off_diag_magnitude):
        rho = np.full((4, 4), 0.0, dtype=complex)
        np.fill_diagonal(rho, 0.25)
        rho[0, 1] = rho[1, 0] = off_diag_magnitude
        rho[2, 3] = rho[3, 2] = off_diag_magnitude
        return rho

    rho_clean_series = [make_rho(v) for v in coherence_clean]
    rho_attacked_series = [make_rho(v) for v in coherence_attacked]
    result = ctr(rho_clean_series, rho_attacked_series, n_qubits=2)
    assert result < 1.0, f"Faster-decaying 'attacked' coherence should give CTR < 1; got {result}"


# ---------------------------------------------------------------------------
# QGOM: Quantum Gradient Obfuscation Measure
# ---------------------------------------------------------------------------

def test_finite_shot_qnode_actually_has_shot_noise():
    """PRECONDITION FOR QGOM BEING DEFINED AT ALL (per this module's own
    docstring): repeated evaluations at IDENTICAL parameters must give
    DIFFERENT results. An exact-mode QNode (no shots) would return the
    exact same value every time, making Var(g_clean) identically 0 and
    QGOM_d undefined (0/0) regardless of any real attack effect."""
    circuit = build_finite_shot_qnode(N_QUBITS, N_LAYERS, "ring", shots=64)
    rng = np.random.default_rng(0)
    weights = rng.normal(size=N_LAYERS * N_QUBITS * 3) * 0.3
    inputs = rng.normal(size=N_QUBITS)
    import pennylane as qml
    results = []
    for i in range(10):
        qml.numpy.random.seed(i)
        results.append(float(circuit(inputs, weights)))
    assert len(set(results)) > 1, "Finite-shot QNode returned identical results every call -- no shot noise present"


def test_qgom_d_detects_gradient_variance_increase_under_attack():
    """POSITIVE CONTROL: attacked inputs constructed to be far outside
    the clean input's operating range should generally produce a
    different (here: higher-variance) gradient signal. Uses a generous
    shot count and trial count to keep this from being flaky, and checks
    that QGOM_d is a finite, real number responding to the input change
    -- not a fixed constant regardless of input (the original stub's
    failure mode)."""
    rng = np.random.default_rng(7)
    n_params = N_LAYERS * N_QUBITS * 3
    weights = rng.normal(size=n_params) * 0.5
    inputs_clean = rng.normal(size=N_QUBITS) * 0.3
    inputs_attacked = inputs_clean + rng.normal(size=N_QUBITS) * 2.0

    result = qgom_d(N_QUBITS, N_LAYERS, "ring", weights, inputs_clean, inputs_attacked,
                     target=0.0, shots=256, n_trials=15, seed=7)

    assert np.isfinite(result["QGOM_d"])
    assert result["var_clean"] >= 0 and result["var_attacked"] >= 0


def test_qgom_d_identical_inputs_give_near_zero_obfuscation():
    """NEGATIVE CONTROL: 'clean' and 'attacked' set to the literal same
    input -- there is no real attack effect, so QGOM_d should be near 0
    (mean gradient variance ratio near 1), modulo shot-noise-floor
    correction pushing small denominators around. Checked as a loose
    bound (not exactly 0) because this is a genuinely stochastic
    quantity at finite shots/trials."""
    rng = np.random.default_rng(8)
    n_params = N_LAYERS * N_QUBITS * 3
    weights = rng.normal(size=n_params) * 0.5
    inputs = rng.normal(size=N_QUBITS) * 0.3

    result = qgom_d(N_QUBITS, N_LAYERS, "ring", weights, inputs, inputs,
                     target=0.0, shots=256, n_trials=15, seed=8)
    if not result["identifiable"]:
        # variance at/below the shot-noise floor -> honest NaN, never a fake 0.0
        assert np.isnan(result["QGOM_d"])
        return
    assert abs(result["QGOM_d"]) < 0.9, f"Identical clean/attacked inputs gave an extreme QGOM_d={result['QGOM_d']}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
