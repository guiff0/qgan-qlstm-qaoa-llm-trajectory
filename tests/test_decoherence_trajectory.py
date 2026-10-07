"""
Tests for src/quantum/decoherence.py's trajectory (Monte Carlo
wavefunction) method -- the scalable alternative to the exact,
density-matrix decoherence_rho_series(), which needs ~16 TB at this
study's primary 20-qubit configuration.

Includes a regression test for a real bug caught while validating this
session: _depolarizing_kraus originally used a different (non-
equivalent) parameterization than qml.DepolarizingChannel, which the
exact method actually calls -- caught because the "clean" condition
(no depolarizing channel involved) matched the exact method to within
statistical noise, but the "attacked" condition (which does use it)
had an error that did NOT shrink with more trajectories -- the
signature of a formula bug, not noise.
"""
import numpy as np
import pennylane as qml
import pytest

from src.quantum.decoherence import (
    _amplitude_damping_kraus, _phase_damping_kraus, _depolarizing_kraus,
    _apply_local_operator, _sample_and_apply_channel,
    decoherence_rho_series, decoherence_rho_series_trajectory,
)
from src.metrics.resilience import ctr, _coherence_measure


def _kraus_matches_pennylane(mine, op):
    """Kraus operators aren't unique (any unitary mixing of a valid set
    is also valid), but all the channels used here have a canonical,
    unambiguous decomposition that PennyLane also uses -- compare
    directly rather than just checking the channel's action."""
    theirs = op.kraus_matrices()
    assert len(mine) == len(theirs)
    for a, b in zip(mine, theirs):
        assert np.allclose(a, b, atol=1e-10)


def test_amplitude_damping_kraus_matches_pennylane():
    _kraus_matches_pennylane(_amplitude_damping_kraus(0.2), qml.AmplitudeDamping(0.2, wires=0))


def test_phase_damping_kraus_matches_pennylane():
    _kraus_matches_pennylane(_phase_damping_kraus(0.3), qml.PhaseDamping(0.3, wires=0))


def test_depolarizing_kraus_matches_pennylane():
    """Regression test for the exact bug found this session."""
    _kraus_matches_pennylane(_depolarizing_kraus(0.15), qml.DepolarizingChannel(0.15, wires=0))


def test_kraus_operators_satisfy_completeness_relation():
    """Sum_i K_i^dagger K_i == I is required for a valid quantum channel
    -- would have caught the depolarizing bug too (the old, wrong
    parameterization also happens to satisfy completeness, so this
    alone isn't sufficient, but it's a necessary sanity check)."""
    for kraus_set in [_amplitude_damping_kraus(0.3), _phase_damping_kraus(0.4), _depolarizing_kraus(0.2)]:
        total = sum(k.conj().T @ k for k in kraus_set)
        assert np.allclose(total, np.eye(2), atol=1e-10)


def test_apply_local_operator_matches_direct_kron_computation():
    """Cross-checks the efficient reshape-based implementation against
    the naive, obviously-correct tensor-product construction."""
    n_qubits = 3
    psi = np.random.default_rng(0).standard_normal(2 ** n_qubits) + 0j
    psi /= np.linalg.norm(psi)
    op = np.array([[0.6, 0.2j], [0.1, 0.8]], dtype=complex)

    for wire in range(n_qubits):
        mats = [np.eye(2, dtype=complex)] * n_qubits
        mats[wire] = op
        full = mats[0]
        for m in mats[1:]:
            full = np.kron(full, m)
        expected = full @ psi
        actual = _apply_local_operator(psi, op, wire, n_qubits)
        assert np.allclose(actual, expected, atol=1e-10)


def test_single_qubit_amplitude_damping_matches_analytical_decay():
    """Population of |1> under pure amplitude damping decays as
    (1-gamma)^n_steps exactly -- a textbook closed-form result."""
    gamma = 0.1
    n_steps = 10
    n_trajectories = 20000
    rng = np.random.default_rng(0)
    kraus = _amplitude_damping_kraus(gamma)
    populations = np.zeros(n_steps)
    for _ in range(n_trajectories):
        psi = np.array([0, 1], dtype=complex)
        for step in range(n_steps):
            psi = _sample_and_apply_channel(psi, kraus, wire=0, n_qubits=1, rng=rng)
            populations[step] += np.abs(psi[1]) ** 2
    populations /= n_trajectories
    analytical = (1 - gamma) ** np.arange(1, n_steps + 1)
    # ~1/sqrt(20000) ~= 0.007 expected statistical noise; allow some margin
    assert np.max(np.abs(populations - analytical)) < 0.02


def test_trajectory_ctr_matches_exact_density_matrix_ctr():
    """The real cross-validation: at a small, tractable qubit count both
    methods can run at, the trajectory method's CTR should converge to
    the exact method's CTR (already validated against the closed-form
    Bell-state case in test_decoherence.py)."""
    n_qubits, n_layers = 4, 2
    weights = np.random.default_rng(0).standard_normal(n_qubits * n_layers * 3) * 0.3
    inputs = np.random.default_rng(1).standard_normal(n_qubits)

    clean_exact = decoherence_rho_series(inputs, weights, n_qubits, n_layers, "ring",
                                          n_steps=10, extra_depolarizing=0.0, seed=0)
    attacked_exact = decoherence_rho_series(inputs, weights, n_qubits, n_layers, "ring",
                                             n_steps=10, extra_depolarizing=0.15, seed=0)
    ctr_exact = ctr(clean_exact, attacked_exact, n_qubits=n_qubits)

    clean_traj = decoherence_rho_series_trajectory(inputs, weights, n_qubits, n_layers, "ring",
                                                     n_steps=10, extra_depolarizing=0.0,
                                                     n_trajectories=1000, seed=42)
    attacked_traj = decoherence_rho_series_trajectory(inputs, weights, n_qubits, n_layers, "ring",
                                                        n_steps=10, extra_depolarizing=0.15,
                                                        n_trajectories=1000, seed=42)
    ctr_traj = ctr(clean_traj, attacked_traj, n_qubits=n_qubits)

    assert abs(ctr_traj - ctr_exact) < 0.02


def test_trajectory_scales_past_the_dense_ceiling():
    """The whole point: must run at a qubit count the exact method
    cannot (check_ctr_tractable would reject this n_qubits for
    decoherence_rho_series)."""
    n_qubits, n_layers = 16, 1  # above MAX_QUBITS_FOR_DENSITY_MATRIX=12
    weights = np.random.default_rng(0).standard_normal(n_qubits * n_layers * 3) * 0.1
    inputs = np.random.default_rng(1).standard_normal(n_qubits)
    series = decoherence_rho_series_trajectory(inputs, weights, n_qubits, n_layers, "ring",
                                                n_steps=3, n_trajectories=5, seed=0,
                                                subsystem=[0, 1])  # small subsystem to keep the test fast
    assert len(series) == 3
    for rho in series:
        assert rho.shape == (4, 4)
        assert np.isclose(np.trace(rho), 1.0, atol=1e-6)
