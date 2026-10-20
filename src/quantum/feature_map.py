"""
Quantum feature map with a matched classical control.

WHY THIS FILE EXISTS
--------------------
In QGAN-LLM the forecast head is a linear layer fitted on real rows; the
quantum generator never reaches it, so forecast RMSE and ASR are identical
to Classical GAN-LLM by construction (Chapter 4, Tables 4D/4F). A test of
quantum advantage in FORECASTING needs the circuit on the inference path.

This module puts it there in the simplest, fairest way: the circuit is a
fixed random feature map phi(x) = <Z_1..Z_n>(x) (same gate sequence as
src/quantum/circuits.py: RY/RZ encoding, RX-RY-RZ layers, CNOT topology),
and a ridge regression is fitted on [x, phi(x)] in closed form. The control
arm replaces phi with a classical random feature map of the SAME output
width and the SAME number of random parameters, applied to the SAME inputs.
Any gain the quantum arm shows over the control is then attributable to the
circuit, not to extra capacity, extra training, or a different head.

The simulator is a plain numpy statevector (no torch, no PennyLane), batched
over rows. Cost is O(rows * 2**n_qubits * gates); keep n_qubits <= ~14 here.
That cost is itself the point: this is a CLASSICAL simulation, so it can
show whether the circuit's function class helps, never a runtime advantage.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Batched statevector simulator (qubit 0 = first tensor axis, as in PennyLane)
# ---------------------------------------------------------------------------
def _rx(theta):
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    U = np.zeros(np.shape(theta) + (2, 2), dtype=np.complex128)
    U[..., 0, 0] = c
    U[..., 1, 1] = c
    U[..., 0, 1] = -1j * s
    U[..., 1, 0] = -1j * s
    return U


def _ry(theta):
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    U = np.zeros(np.shape(theta) + (2, 2), dtype=np.complex128)
    U[..., 0, 0] = c
    U[..., 1, 1] = c
    U[..., 0, 1] = -s
    U[..., 1, 0] = s
    return U


def _rz(theta):
    U = np.zeros(np.shape(theta) + (2, 2), dtype=np.complex128)
    U[..., 0, 0] = np.exp(-0.5j * theta)
    U[..., 1, 1] = np.exp(0.5j * theta)
    return U


def _apply_1q(psi: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
    """psi: (B, 2, ..., 2); U: (2,2) shared or (B,2,2) per row; qubit q."""
    moved = np.moveaxis(psi, q + 1, -1)               # (B, ..., 2)
    if U.ndim == 2:
        out = moved @ U.T
    else:
        out = np.einsum("b...j,bij->b...i", moved, U)
    return np.moveaxis(out, -1, q + 1)


def _apply_cnot(psi: np.ndarray, control: int, target: int) -> np.ndarray:
    idx = [slice(None)] * psi.ndim
    idx[control + 1] = 1
    sub = psi[tuple(idx)]                              # control axis removed
    t_axis = target + 1 - (1 if target > control else 0)
    psi = psi.copy()
    psi[tuple(idx)] = np.flip(sub, axis=t_axis).copy()
    return psi


def _entangle(psi: np.ndarray, n: int, topology: str) -> np.ndarray:
    if topology == "ring":
        for q in range(n):
            psi = _apply_cnot(psi, q, (q + 1) % n)
    elif topology == "linear":
        for q in range(n - 1):
            psi = _apply_cnot(psi, q, q + 1)
    elif topology == "full":
        for i in range(n):
            for j in range(i + 1, n):
                psi = _apply_cnot(psi, i, j)
    elif topology == "none":
        pass
    else:
        raise ValueError(f"unknown entanglement topology {topology!r}")
    return psi


def circuit_expvals(inputs: np.ndarray, weights: np.ndarray, n_qubits: int,
                    n_layers: int, entanglement: str = "ring") -> np.ndarray:
    """<Z_i> for every qubit. Same gate sequence as circuits.build_qlstm_qnode.

    inputs : (B, >=n_qubits) rotation angles (first n_qubits columns used)
    weights: (n_layers * n_qubits * 3,)
    returns: (B, n_qubits) in [-1, 1]
    """
    inputs = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.size != n_layers * n_qubits * 3:
        raise ValueError(f"expected {n_layers * n_qubits * 3} weights, got {weights.size}")
    if inputs.shape[1] < n_qubits:
        raise ValueError(f"need at least {n_qubits} input columns, got {inputs.shape[1]}")
    B = inputs.shape[0]
    psi = np.zeros((B,) + (2,) * n_qubits, dtype=np.complex128)
    psi[(slice(None),) + (0,) * n_qubits] = 1.0

    for i in range(n_qubits):                          # encoding
        psi = _apply_1q(psi, _ry(inputs[:, i]), i)
        psi = _apply_1q(psi, _rz(inputs[:, i] * 0.1), i)
    for layer in range(n_layers):                      # variational layers
        base = layer * n_qubits * 3
        for q in range(n_qubits):
            psi = _apply_1q(psi, _rx(weights[base + 3 * q]), q)
            psi = _apply_1q(psi, _ry(weights[base + 3 * q + 1]), q)
            psi = _apply_1q(psi, _rz(weights[base + 3 * q + 2]), q)
        psi = _entangle(psi, n_qubits, entanglement)

    probs = (psi.real ** 2 + psi.imag ** 2).reshape(B, -1)
    out = np.empty((B, n_qubits))
    for i in range(n_qubits):
        p = probs.reshape(B, 2 ** i, 2, 2 ** (n_qubits - i - 1)).sum(axis=(1, 3))
        out[:, i] = p[:, 0] - p[:, 1]
    return out


# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------
class QuantumFeatureMap:
    """Fixed random-parameter PQC feature map. Output width = n_qubits."""

    name = "quantum"

    def __init__(self, n_qubits: int = 8, n_layers: int = 3, entanglement: str = "ring",
                 seed: int = 0, chunk: int = 20_000):
        if n_qubits < 2 and entanglement != "none":
            raise ValueError("entanglement needs >= 2 qubits")
        self.n_qubits, self.n_layers, self.entanglement = n_qubits, n_layers, entanglement
        self.chunk = chunk
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(0.0, 2.0 * np.pi, size=n_layers * n_qubits * 3)

    @property
    def n_random_params(self) -> int:
        return int(self.weights.size)

    @property
    def n_outputs(self) -> int:
        return self.n_qubits

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Z: (N, >=n_qubits) already-scaled angles."""
        Z = np.asarray(Z, dtype=np.float64)
        out = np.empty((len(Z), self.n_qubits))
        for s in range(0, len(Z), self.chunk):
            out[s:s + self.chunk] = circuit_expvals(
                Z[s:s + self.chunk], self.weights, self.n_qubits, self.n_layers, self.entanglement)
        return out


class ClassicalRandomFeatureMap:
    """Matched control: tanh(Z[:, :n] @ W + b) with n_outputs == n_qubits and
    the same NUMBER of random parameters as the circuit (W, b are filled up to
    that count; surplus parameters are used as a second random layer so the
    parameter budget is spent, not ignored)."""

    name = "classical_rf"

    def __init__(self, n_inputs: int = 8, n_outputs: int = 8, n_random_params: int = 72,
                 seed: int = 0):
        self.n_inputs, self.n_out = n_inputs, n_outputs
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 1.0, size=(n_inputs, n_outputs))
        self.b1 = rng.uniform(0.0, 2.0 * np.pi, size=n_outputs)
        used = self.W1.size + self.b1.size
        surplus = max(0, int(n_random_params) - used)
        # second random layer (n_outputs x k) consumes the remaining budget
        k = min(n_outputs, surplus // max(n_outputs, 1))
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(max(n_outputs, 1)), size=(n_outputs, k)) if k else None
        self._n_random_params = int(used + (self.W2.size if self.W2 is not None else 0))

    @property
    def n_random_params(self) -> int:
        return self._n_random_params

    @property
    def n_outputs(self) -> int:
        return self.n_out

    def transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)[:, : self.n_inputs]
        h = np.tanh(Z @ self.W1 + self.b1)
        if self.W2 is not None:
            k = self.W2.shape[1]
            h = h.copy()
            h[:, :k] = np.tanh(h @ self.W2)
        return h
