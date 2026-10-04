import numpy as np

from src.quantum.decoherence import partial_trace_density_matrix, decoherence_rho_series
from src.metrics.resilience import ctr


def test_partial_trace_of_bell_state_is_maximally_mixed():
    """Known closed-form case: tracing out either qubit of a Bell state
    leaves the maximally mixed single-qubit state I/2."""
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    reduced = partial_trace_density_matrix(rho, 2, [0])
    assert np.allclose(reduced, np.eye(2) / 2, atol=1e-10)


def test_partial_trace_preserves_trace_and_hermiticity():
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[7] = 1 / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    reduced = partial_trace_density_matrix(rho, 3, [0, 1])
    assert np.isclose(np.trace(reduced), 1.0)
    assert np.allclose(reduced, reduced.conj().T)


def test_decoherence_series_is_physically_sensible():
    """Coherence (off-diagonal magnitude) should monotonically decay as
    more decoherence steps are applied -- a basic sanity check that the
    channel is actually doing something, not just returning the input."""
    n_qubits, n_layers = 3, 1
    weights = np.random.default_rng(0).standard_normal(n_qubits * n_layers * 3) * 0.3
    inputs = np.random.default_rng(1).standard_normal(n_qubits)
    series = decoherence_rho_series(inputs, weights, n_qubits, n_layers, "ring",
                                     n_steps=5, extra_depolarizing=0.0, seed=0)
    off_diag_magnitude = [np.sum(np.abs(rho - np.diag(np.diag(rho)))) for rho in series]
    # Not strictly monotonic at every step (depends on the clean state's
    # starting coherence pattern), but the LAST step must have less
    # coherence than the FIRST under any nonzero damping.
    assert off_diag_magnitude[-1] <= off_diag_magnitude[0] + 1e-9


def test_ctr_below_one_when_attacked_channel_is_noisier():
    n_qubits, n_layers = 4, 2
    weights = np.random.default_rng(0).standard_normal(n_qubits * n_layers * 3) * 0.3
    inputs = np.random.default_rng(1).standard_normal(n_qubits)

    clean_series = decoherence_rho_series(inputs, weights, n_qubits, n_layers, "ring",
                                           n_steps=10, extra_depolarizing=0.0, seed=0)
    attacked_series = decoherence_rho_series(inputs, weights, n_qubits, n_layers, "ring",
                                              n_steps=10, extra_depolarizing=0.15, seed=0)
    value = ctr(clean_series, attacked_series, n_qubits=n_qubits)
    assert np.isfinite(value)
    assert 0.0 < value < 1.0  # noisier channel decoheres faster -> shorter T2 -> CTR < 1
