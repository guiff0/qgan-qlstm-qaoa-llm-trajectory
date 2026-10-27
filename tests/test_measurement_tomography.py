import numpy as np
import pennylane as qml
from src.quantum.tomography import (measurement_tomography, pauli_expectations, reconstruct_density_matrix,
                                    purity, von_neumann_entropy)


def _bell(_p):
    qml.Hadamard(wires=0); qml.CNOT(wires=[0, 1])


def _product(p):
    qml.RY(p[0], wires=0); qml.RY(p[1], wires=1); qml.CNOT(wires=[1, 2])


def test_exact_expectations_reconstruct_bell_marginal():
    exp = pauli_expectations(_bell, None, 2, [0], shots=None)
    rho = reconstruct_density_matrix(exp, 1)
    assert np.allclose(rho, np.eye(2) / 2, atol=1e-9)
    assert abs(von_neumann_entropy(rho) - 1.0) < 1e-6 and abs(purity(rho) - 0.5) < 1e-6


def test_finite_shot_tomography_close_to_exact():
    p = np.array([0.7, 1.9])
    r = measurement_tomography(_product, p, 3, [0, 1], shots=20000, seed=1)
    assert r["n_pauli_words"] == 15
    assert r["fidelity_to_exact"] > 0.98 and r["trace_distance"] < 0.05
    assert abs(np.trace(r["rho_hat"]) - 1) < 1e-9 and np.linalg.eigvalsh(r["rho_hat"]).min() > -1e-9


def test_rejects_large_subsystems():
    try:
        pauli_expectations(_bell, None, 2, [0, 1, 2, 3, 4], shots=None)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
