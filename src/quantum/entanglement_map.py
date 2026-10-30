"""
Pairwise entanglement map: quantum mutual information between every pair of qubits.

    I(i:j) = S(i) + S(j) - S(ij)      (bits; S = von Neumann entropy of the reduced state)

For a PURE global state (the generator statevector) S(i) is the entanglement entropy of qubit i with
all the others, and I(i:j) measures the total (classical + quantum) correlation between i and j.
It is 0 for a product state, 2 for a Bell pair, 1 for every pair of a GHZ state. It is symmetric, so
the map is a symmetric N x N matrix with the diagonal set to 0 (single-qubit entropies are returned
separately).

numpy-only on purpose (no PennyLane import) so it can be tested anywhere. The statevector comes from
src/quantum/tomography.py; as everywhere in this project it is an exact SIMULATOR quantity -- a real
device gives no access to the statevector.
"""
from __future__ import annotations

import json
import os

import numpy as np


def _entropy_bits(rho: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return float(-(ev * np.log2(ev)).sum())


def pairwise_mutual_information(state: np.ndarray, n_qubits: int) -> dict:
    """Returns {"mi": (N,N) matrix in bits, "single_entropy": (N,) in bits}."""
    state = np.asarray(state, dtype=complex).reshape(-1)
    if state.size != 2 ** n_qubits:
        raise ValueError(f"statevector has {state.size} amplitudes, expected 2**{n_qubits}")
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=1e-6):
        raise ValueError(f"statevector is not normalised (norm={norm:.6f})")
    psi = state.reshape([2] * n_qubits)

    two = {}                       # (i, j) -> 4x4 reduced density matrix
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            m = np.moveaxis(psi, [i, j], [0, 1]).reshape(4, -1)
            two[(i, j)] = m @ m.conj().T

    single = np.zeros(n_qubits)
    for i in range(n_qubits):
        if n_qubits == 1:
            single[i] = 0.0
            continue
        j = i + 1 if i + 1 < n_qubits else i - 1
        rho2 = two[(min(i, j), max(i, j))].reshape(2, 2, 2, 2)
        rho_i = np.einsum("abcb->ac", rho2) if i < j else np.einsum("abad->bd", rho2)
        single[i] = _entropy_bits(rho_i)

    mi = np.zeros((n_qubits, n_qubits))
    for (i, j), rho in two.items():
        mi[i, j] = mi[j, i] = max(0.0, single[i] + single[j] - _entropy_bits(rho))
    return {"mi": mi, "single_entropy": single}


def save_entanglement_map(path: str, mi: np.ndarray, single_entropy: np.ndarray, meta: dict) -> str:
    """Writes the map as JSON (archiving any previous file at the same path first)."""
    try:
        from ..utils.run_identity import archive_file
        archive_file(path)
    except Exception:
        pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {**meta,
               "n_qubits": int(mi.shape[0]),
               "unit": "bits",
               "mean_pairwise_mi": float(mi[np.triu_indices_from(mi, k=1)].mean()) if mi.shape[0] > 1 else 0.0,
               "max_pairwise_mi": float(mi.max()),
               "single_qubit_entropy": single_entropy.tolist(),
               "mutual_information": mi.tolist(),
               "note": ("Exact simulator statevector, fixed reference input; no density matrix is "
                        "reconstructed from measurements.")}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
