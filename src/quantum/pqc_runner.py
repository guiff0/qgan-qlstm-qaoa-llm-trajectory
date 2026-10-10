"""
Torch-free runner for the QLSTM generator's parameterized circuit that
computes the *vector-Jacobian product* (VJP) directly.

WHY THIS EXISTS (measured, see README_FIX.md):
PennyLane's `diff_method="adjoint"` on a QNode that returns 20 separate
<Z_k> expectation values computes the FULL Jacobian: one adjoint sweep per
observable. At 20 qubits / 4 layers that is ~28 s per sample (single core).
But training never needs the full Jacobian -- the generator's loss only
needs  J^T g  for one upstream-gradient vector g (the gradient flowing back
from the output Linear layer). J^T g is the gradient of ONE observable,
H_g = sum_k g_k Z_k, so a single adjoint sweep (~1.7 s per sample) gives
exactly the same numbers. Same math, ~16x cheaper.

Forward-only evaluation (no gradients) uses diff_method=None (~0.27 s per
sample at 20 qubits), so generating synthetic data / the discriminator's
fake batches never pays for gradients at all.

This module imports only numpy + pennylane so it can be unit-tested without
torch (tests/test_pqc_runner.py).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pennylane as qml
from autograd import grad as _autograd_grad
from pennylane import numpy as pnp


def _apply_ansatz(inputs, weights, n_qubits: int, n_layers: int, entanglement: str) -> None:
    """Identical gate sequence to circuits.build_qlstm_qnode (encoding layer,
    then n_layers x [RX RY RZ per qubit + entangling CNOTs])."""
    for i in range(min(n_qubits, len(inputs))):
        qml.RY(inputs[i], wires=i)
        qml.RZ(inputs[i] * 0.1, wires=i)

    n_params_per_layer = n_qubits * 3
    for layer in range(n_layers):
        start = layer * n_params_per_layer
        for q in range(n_qubits):
            idx = start + q * 3
            qml.RX(weights[idx], wires=q)
            qml.RY(weights[idx + 1], wires=q)
            qml.RZ(weights[idx + 2], wires=q)

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
        else:
            raise ValueError(f"Unknown entanglement topology: {entanglement}")


def _linear_combination(coeffs, observables):
    cls = getattr(getattr(qml, "ops", None), "LinearCombination", None) or qml.Hamiltonian
    return cls(coeffs, observables)


class PQCRunner:
    def __init__(self, n_qubits: int, n_layers: int, entanglement: str,
                 dev_name: str = "lightning.qubit"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.dev_name = dev_name
        nq, nl, ent = n_qubits, n_layers, entanglement

        fwd_dev = qml.device(dev_name, wires=nq)

        @qml.qnode(fwd_dev, diff_method=None)
        def _forward(inputs, weights):
            _apply_ansatz(inputs, weights, nq, nl, ent)
            return [qml.expval(qml.PauliZ(i)) for i in range(nq)]

        grad_dev = qml.device(dev_name, wires=nq)
        # adjoint needs an analytic statevector simulator; fall back to
        # parameter-shift for anything else (e.g. real-QPU backends).
        diff_method = "adjoint" if dev_name.startswith("lightning") or dev_name == "default.qubit" else "parameter-shift"

        @qml.qnode(grad_dev, diff_method=diff_method)
        def _weighted(inputs, weights, g):
            _apply_ansatz(inputs, weights, nq, nl, ent)
            return qml.expval(_linear_combination(g, [qml.PauliZ(i) for i in range(nq)]))

        self._forward = _forward
        self._weighted = _weighted

    # ---- forward only ---------------------------------------------------
    def forward(self, inputs: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """inputs: (B, n_qubits); theta: (n_params,). Returns (B, n_qubits)."""
        out = np.empty((inputs.shape[0], self.n_qubits), dtype=np.float64)
        for b in range(inputs.shape[0]):
            out[b] = np.asarray(self._forward(inputs[b], theta), dtype=np.float64)
        return out

    # ---- vector-Jacobian product ---------------------------------------
    def vjp(self, inputs: np.ndarray, theta: np.ndarray, grad_out: np.ndarray,
            need_input_grad: bool = False) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """grad_out: (B, n_qubits) upstream gradient dL/d<Z_k>. Returns
        (dL/d inputs or None, dL/d theta summed over the batch)."""
        grad_theta = np.zeros_like(theta, dtype=np.float64)
        grad_inputs = np.zeros(inputs.shape, dtype=np.float64) if need_input_grad else None
        for b in range(inputs.shape[0]):
            g = np.asarray(grad_out[b], dtype=np.float64)
            if not np.any(g):
                continue
            nq = self.n_qubits
            if need_input_grad:
                vec = pnp.array(np.concatenate([inputs[b], theta]), requires_grad=True)
                full = np.asarray(_autograd_grad(lambda v: self._weighted(v[:nq], v[nq:], g))(vec))
                grad_inputs[b] = full[:nq]
                gt = full[nq:]
            else:
                x = pnp.array(inputs[b], requires_grad=False)
                th = pnp.array(theta, requires_grad=True)
                gt = _autograd_grad(lambda t: self._weighted(x, t, g))(th)
            grad_theta += np.asarray(gt)
        return grad_inputs, grad_theta
