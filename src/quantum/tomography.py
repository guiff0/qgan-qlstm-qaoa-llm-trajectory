"""
Quantum state tomography metrics: entanglement entropy, purity, and
state fidelity.

THIS REPLACES THE ORIGINAL STUB:

    def _measure_entanglement(self):
        return np.random.uniform(2.5, 4.0)  # Placeholder

That stub returned a random number labeled as "entanglement entropy."
Any correlation the dissertation reports between this quantity and
detection accuracy (e.g. r=0.67) would have been a correlation with
noise, not with anything the model actually did — this is exactly
the kind of self-confirming result flagged earlier. Below is a real,
literature-standard calculation:

- Get the full statevector from the PennyLane circuit
  (qml.state(), not qml.expval — expval alone cannot reconstruct
  entanglement, which is a property of the joint state).
- Partition the qubits into two halves (or a caller-specified cut).
- Compute the reduced density matrix of one half via partial trace.
- Von Neumann entropy of that reduced density matrix IS the
  entanglement entropy between the two halves for a pure global state.
- Purity = Tr(rho^2), a companion metric already named in the
  dissertation's Table 49 (Quantum State Tomography Metrics).
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def statevector_from_circuit(circuit_fn, params, n_qubits: int, dev_name: str = "default.qubit"):
    """
    Run `circuit_fn(params)` (a function that only applies gates, no
    measurement) on a fresh device configured to return the full
    statevector, and return that state as a numpy array of length 2**n_qubits.
    """
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev)
    def _state_qnode(p):
        circuit_fn(p)
        return qml.state()

    return np.asarray(_state_qnode(params))


def reduced_density_matrix(statevector: np.ndarray, n_qubits: int, subsystem_qubits: list[int]) -> np.ndarray:
    """
    Partial trace of |psi><psi| over the complement of `subsystem_qubits`,
    returning the reduced density matrix of the requested subsystem.
    """
    psi = statevector.reshape([2] * n_qubits)
    n_sub = len(subsystem_qubits)
    complement = [q for q in range(n_qubits) if q not in subsystem_qubits]

    # Move subsystem qubits to the front, complement to the back
    order = subsystem_qubits + complement
    psi = np.transpose(psi, axes=order)
    psi = psi.reshape(2 ** n_sub, 2 ** len(complement))

    rho = psi @ psi.conj().T  # (2^n_sub, 2^n_sub)
    return rho


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    """S(rho) = -Tr(rho log2 rho), computed via eigenvalues for numerical stability."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals.real, eps, 1.0)
    return float(-np.sum(eigvals * np.log2(eigvals)))


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2). Purity = 1 for a pure state, 1/d for maximally mixed of dimension d."""
    return float(np.real(np.trace(rho @ rho)))


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Uhlmann fidelity between two density matrices. For two pure states
    reduced to the same subsystem this reduces to |<phi|psi>|^2.
    Used to compare the QGAN's synthetic-data-encoding state against
    a reference/target state if one is defined.
    """
    # sqrt(rho) via eigendecomposition (rho is Hermitian PSD)
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals.real, 0, None)
    sqrt_rho = (evecs * np.sqrt(evals)) @ evecs.conj().T

    inner = sqrt_rho @ sigma @ sqrt_rho
    evals_inner = np.linalg.eigvalsh(inner)
    evals_inner = np.clip(evals_inner.real, 0, None)
    return float(np.sum(np.sqrt(evals_inner)) ** 2)


def entanglement_metrics(circuit_fn, params, n_qubits: int,
                          dev_name: str = "default.qubit",
                          cut: int | None = None) -> dict:
    """
    Full tomography report for one circuit execution:
      - entanglement_entropy: von Neumann entropy across the bipartition
      - purity: purity of the same reduced subsystem
      - subsystem_size: how many qubits were on each side of the cut

    `cut` is the number of qubits assigned to subsystem A; defaults to
    an even bipartition (n_qubits // 2), which is the standard choice
    for reporting a single scalar entanglement-entropy value per circuit.
    """
    if cut is None:
        cut = n_qubits // 2
    subsystem_qubits = list(range(cut))

    state = statevector_from_circuit(circuit_fn, params, n_qubits, dev_name)
    rho_A = reduced_density_matrix(state, n_qubits, subsystem_qubits)

    return {
        "entanglement_entropy": von_neumann_entropy(rho_A),
        "purity": purity(rho_A),
        "subsystem_size": cut,
        "max_possible_entropy": float(cut),  # log2(2^cut) — for normalizing if needed
    }


# ---------------------------------------------------------------------------
# Measurement-based tomography of a small subsystem
#
# entanglement_metrics() above reads the exact statevector of a SIMULATOR and
# takes a partial trace -- an exact calculation, not tomography (a real device
# gives no access to the statevector). The functions below do what Ch.3 calls
# quantum state tomography: estimate Pauli expectation values from a finite
# number of measurement shots on the subsystem and rebuild its density matrix.
# The cost is 4^k Pauli words for a k-qubit subsystem, so k is capped at 4.
# ---------------------------------------------------------------------------
import itertools

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
MAX_TOMOGRAPHY_QUBITS = 4


def _pauli_matrix(word: str) -> np.ndarray:
    m = np.array([[1.0 + 0j]])
    for ch in word:
        m = np.kron(m, _PAULI[ch])
    return m


def pauli_expectations(circuit_fn, params, n_qubits: int, subsystem_qubits: list[int],
                       shots: int | None, dev_name: str = "default.qubit", seed: int = 0) -> dict:
    """Estimate <P> for every Pauli word P on `subsystem_qubits` (identity excluded).
    shots=None gives exact expectation values; an int gives a finite-shot estimate."""
    k = len(subsystem_qubits)
    if k < 1 or k > MAX_TOMOGRAPHY_QUBITS:
        raise ValueError(f"subsystem must have 1..{MAX_TOMOGRAPHY_QUBITS} qubits, got {k}")
    dev = qml.device(dev_name, wires=n_qubits, shots=shots, seed=seed) if shots is not None \
        else qml.device(dev_name, wires=n_qubits)
    ops = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    out = {}
    for word in ("".join(w) for w in itertools.product("IXYZ", repeat=k)):
        if set(word) == {"I"}:
            continue
        obs = None
        for ch, wire in zip(word, subsystem_qubits):
            if ch == "I":
                continue
            term = ops[ch](wire)
            obs = term if obs is None else obs @ term

        @qml.qnode(dev)
        def _q(p, _obs=obs):
            circuit_fn(p)
            return qml.expval(_obs)

        out[word] = float(_q(params))
    return out


def reconstruct_density_matrix(expectations: dict, k: int, physical: bool = True) -> np.ndarray:
    """Linear-inversion estimate rho = 2^-k * sum_P <P> P, optionally projected onto
    the nearest valid state (clip negative eigenvalues, renormalise to trace 1)."""
    rho = np.eye(2 ** k, dtype=complex) / 2 ** k
    for word, e in expectations.items():
        rho = rho + e * _pauli_matrix(word) / 2 ** k
    if physical:
        rho = (rho + rho.conj().T) / 2
        vals, vecs = np.linalg.eigh(rho)
        vals = np.clip(vals, 0, None)
        vals = vals / vals.sum()
        rho = (vecs * vals) @ vecs.conj().T
    return rho


def measurement_tomography(circuit_fn, params, n_qubits: int, subsystem_qubits: list[int],
                           shots: int = 4096, dev_name: str = "default.qubit", seed: int = 0,
                           compare_exact: bool = True) -> dict:
    """Shot-based tomography of `subsystem_qubits`. When compare_exact is True the
    reconstruction is scored against the simulator's exact reduced state, which is
    the only reason a statevector is touched (not available on hardware)."""
    k = len(subsystem_qubits)
    exp = pauli_expectations(circuit_fn, params, n_qubits, subsystem_qubits, shots, dev_name, seed)
    rho = reconstruct_density_matrix(exp, k)
    res = {"subsystem_qubits": list(subsystem_qubits), "shots_per_pauli_word": shots,
           "n_pauli_words": len(exp), "rho_hat": rho,
           "entropy_hat": von_neumann_entropy(rho), "purity_hat": purity(rho)}
    if compare_exact:
        state = statevector_from_circuit(circuit_fn, params, n_qubits, dev_name)
        rho_true = reduced_density_matrix(state, n_qubits, list(subsystem_qubits))
        res.update(rho_exact=rho_true, fidelity_to_exact=state_fidelity(rho, rho_true),
                   entropy_exact=von_neumann_entropy(rho_true), purity_exact=purity(rho_true),
                   trace_distance=float(0.5 * np.abs(np.linalg.eigvalsh(rho - rho_true)).sum()))
    return res


def save_pairwise_entanglement_map(circuit_fn, params, n_qubits: int, out_path: str, meta: dict,
                                   dev_name: str = "default.qubit") -> str:
    """Pairwise mutual-information map of the circuit's output state (Figure 32 of the dissertation),
    written to `out_path` as JSON. Costs one statevector + n(n-1)/2 small partial traces."""
    from .entanglement_map import pairwise_mutual_information, save_entanglement_map
    state = statevector_from_circuit(circuit_fn, params, n_qubits, dev_name)
    res = pairwise_mutual_information(state, n_qubits)
    return save_entanglement_map(out_path, res["mi"], res["single_entropy"], meta)
