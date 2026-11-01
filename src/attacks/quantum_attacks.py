"""
Quantum-circuit-targeted attacks: Quantum Noise Injection (-> QSFR) and
Entanglement Disruption (-> EER), from the paper's "Quantum-Specific
Attack Vectors" table (Table 6 companion).

THESE ARE A DIFFERENT THREAT MODEL FROM src/attacks/adversarial.py.
FGSM/PGD/CW there perturb the CLASSICAL features fed into the pipeline
-- the standard adversarial-ML setup behind H2/H4's ASR. The two
attacks here instead perturb the QUANTUM CIRCUIT itself (its trained
rotation parameters, or the structure right after its entangling
gates), modeling an attacker with a fundamentally different capability
-- e.g. a compromised quantum-cloud backend silently drifting gate
calibration, not an attacker manipulating market-data inputs. Report
these as a distinct attack surface in the paper, not as "the quantum
version of the same attack."

Both attacks reuse src/quantum/tomography.py's partial-trace/von
Neumann machinery rather than reimplementing it -- see that file for
why those specific calculations are correct.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

from ..quantum.tomography import (
    statevector_from_circuit,
    reduced_density_matrix,
    von_neumann_entropy,
    state_fidelity,
)


def _apply_entangling(n_qubits: int, entanglement: str) -> None:
    """Identical topology logic to circuits.py -- kept here as its own
    function (rather than imported) because entanglement_disruption_attack
    below needs to insert gates immediately after this call, which
    circuits.py's existing functions don't expose a hook for."""
    if entanglement == "ring":
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
    elif entanglement == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
    elif entanglement == "linear":
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
    elif entanglement == "none":
        pass  # product-state control: no entangling gates (zero-entanglement sweep level)
    else:
        raise ValueError(f"Unknown entanglement topology: {entanglement}")


def _clean_gate_sequence(inputs_np: np.ndarray, n_qubits: int, n_layers: int, entanglement: str):
    """Returns a function of `weights` reproducing circuits.py's exact
    encoding + variational + entangling sequence, with no attack applied.
    Keep in sync with circuits.build_qlstm_qnode if that architecture
    changes -- this is deliberately a duplicate, not an import, because
    the disruption variant below needs a hook mid-sequence that the
    shared circuits.py helpers don't provide."""
    def _apply(weights):
        for i in range(min(n_qubits, len(inputs_np))):
            qml.RY(inputs_np[i], wires=i)
            qml.RZ(inputs_np[i] * 0.1, wires=i)

        n_params_per_layer = n_qubits * 3
        for layer in range(n_layers):
            start = layer * n_params_per_layer
            for q in range(n_qubits):
                idx = start + q * 3
                qml.RX(weights[idx], wires=q)
                qml.RY(weights[idx + 1], wires=q)
                qml.RZ(weights[idx + 2], wires=q)
            _apply_entangling(n_qubits, entanglement)

    return _apply


def quantum_noise_injection_attack(inputs_np: np.ndarray, weights_np: np.ndarray,
                                    n_qubits: int, n_layers: int, entanglement: str,
                                    epsilon: float = 0.3, dev_name: str = "default.qubit",
                                    cut: int | None = None, seed: int = 0) -> dict:
    """
    Attacker capability: inject Gaussian noise directly into the trained
    circuit's rotation parameters before execution (white-box access to
    gate calibration).

    Returns QSFR = fidelity(rho_clean, rho_attacked). Table 6 writes
    this as F_attacked / F_clean; F_clean is the state's self-fidelity
    (trivially 1), so QSFR reduces to a single fidelity computation
    between the clean and attacked reduced states.
    """
    if cut is None:
        cut = n_qubits // 2
    subsystem = list(range(cut))
    rng = np.random.default_rng(seed)

    clean_fn = _clean_gate_sequence(inputs_np, n_qubits, n_layers, entanglement)
    state_clean = statevector_from_circuit(clean_fn, weights_np, n_qubits, dev_name)
    rho_clean = reduced_density_matrix(state_clean, n_qubits, subsystem)

    weights_attacked = weights_np + rng.normal(0, epsilon, size=weights_np.shape)
    state_attacked = statevector_from_circuit(clean_fn, weights_attacked, n_qubits, dev_name)
    rho_attacked = reduced_density_matrix(state_attacked, n_qubits, subsystem)

    return {
        "qsfr": state_fidelity(rho_clean, rho_attacked),
        "epsilon": epsilon,
        "subsystem_size": cut,
    }


def entanglement_disruption_attack(inputs_np: np.ndarray, weights_np: np.ndarray,
                                    n_qubits: int, n_layers: int, entanglement: str,
                                    epsilon: float = 0.3, dev_name: str = "default.qubit",
                                    cut: int | None = None, seed: int = 0) -> dict:
    """
    Attacker capability: cannot see or alter the trained rotation
    parameters, but can inject a small random single-qubit rotation on
    every qubit immediately after each entangling layer -- perturbing
    exactly the correlations those CNOTs just created, leaving the
    encoding and the trained variational rotations themselves untouched.
    This is what distinguishes it from quantum_noise_injection_attack
    above (whole-circuit parameter noise): this attack targets
    entanglement structure specifically.

    Returns EER = entropy(attacked) / entropy(clean).
    """
    if cut is None:
        cut = n_qubits // 2
    subsystem = list(range(cut))
    rng = np.random.default_rng(seed)

    clean_fn = _clean_gate_sequence(inputs_np, n_qubits, n_layers, entanglement)
    state_clean = statevector_from_circuit(clean_fn, weights_np, n_qubits, dev_name)
    rho_clean = reduced_density_matrix(state_clean, n_qubits, subsystem)
    s_clean = von_neumann_entropy(rho_clean)

    disruption_angles = rng.normal(0, epsilon, size=n_qubits * n_layers)

    def _disrupted_apply(weights):
        for i in range(min(n_qubits, len(inputs_np))):
            qml.RY(inputs_np[i], wires=i)
            qml.RZ(inputs_np[i] * 0.1, wires=i)

        n_params_per_layer = n_qubits * 3
        for layer in range(n_layers):
            start = layer * n_params_per_layer
            for q in range(n_qubits):
                idx = start + q * 3
                qml.RX(weights[idx], wires=q)
                qml.RY(weights[idx + 1], wires=q)
                qml.RZ(weights[idx + 2], wires=q)
            _apply_entangling(n_qubits, entanglement)
            # THE ATTACK -- disrupt correlations this layer's entangling
            # gates just created, before the next layer builds on them.
            for q in range(n_qubits):
                qml.RZ(disruption_angles[layer * n_qubits + q], wires=q)

    state_attacked = statevector_from_circuit(_disrupted_apply, weights_np, n_qubits, dev_name)
    rho_attacked = reduced_density_matrix(state_attacked, n_qubits, subsystem)
    s_attacked = von_neumann_entropy(rho_attacked)

    eer = s_attacked / s_clean if s_clean > 1e-9 else float("nan")
    return {
        "eer": eer,
        "entropy_clean": s_clean,
        "entropy_attacked": s_attacked,
        "epsilon": epsilon,
        "subsystem_size": cut,
    }
