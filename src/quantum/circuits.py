"""
QLSTM generator circuit.

*** CRITICAL BUG FOUND AND FIXED IN THIS FILE ***

The original QLSTMGenerator.forward() did this, per sample in a Python loop:

    inputs = z[i].detach().numpy()
    weights = self.theta.detach().numpy()
    result = self._quantum_circuit(inputs, weights)

`.detach().numpy()` strips both `z` and `self.theta` out of the PyTorch
autograd graph before the quantum circuit ever runs. That means when
`total_loss.backward()` was later called during training, no gradient
could reach `self.theta` — the circuit's own trainable parameters.
The generator's *classical* output_layer would still learn something
(it sees fixed, un-trainable quantum outputs as input), but the quantum
circuit itself would train to nothing but its random initialization.
Every result attributed to "20-qubit ring-entangled QGAN generator" vs.
"12-qubit" vs. "linear topology" would, in the original code, actually
just be measuring different random-but-frozen quantum embeddings, not
a trained effect of entanglement structure or qubit count.

THE FIX: use PennyLane's native torch interface (`interface="torch"`)
so the QNode is a differentiable PyTorch operation, and never call
.detach() or .numpy() on parameters that need gradients.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from .pqc_runner import PQCRunner


def build_qlstm_qnode(n_qubits: int, n_layers: int, entanglement: str,
                       dev_name: str = "default.qubit"):
    """
    Returns a torch-differentiable QNode implementing the encoding +
    variational-layer + measurement structure described in the
    dissertation's Materials/Instrumentation section.
    """
    dev = qml.device(dev_name, wires=n_qubits)

    # default.qubit: full backprop through the simulator (fast, but a single
    #   gate application broadcasts to a (batch, 2, 2, ..., 2) tensor via
    #   einsum -- at n_qubits=20/batch=64 that's exactly the 1GB allocation
    #   that OOM'd).
    # lightning.*: no native batch broadcasting, so PennyLane expands the
    #   batch into 64 separate single-sample executions automatically --
    #   each only needs one 2**20-amplitude statevector (~16MB), which is
    #   what actually fixes the memory blowup. Prefer "adjoint" here: one
    #   extra pass per sample, vs. ~2*n_params (=480 for this config) passes
    #   for parameter-shift.
    # anything else (e.g. a real QPU backend): fall back to parameter-shift,
    #   since adjoint requires an analytic statevector simulator.
    if dev_name == "default.qubit":
        diff_method = "backprop"
    elif dev_name.startswith("lightning"):
        diff_method = "adjoint"
    else:
        diff_method = "parameter-shift"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # --- Encoding layer ---
        for i in range(min(n_qubits, inputs.shape[-1])):
            qml.RY(inputs[..., i], wires=i)
            qml.RZ(inputs[..., i] * 0.1, wires=i)

        # --- Variational layers ---
        n_params_per_layer = n_qubits * 3
        for layer in range(n_layers):
            start = layer * n_params_per_layer
            for q in range(n_qubits):
                idx = start + q * 3
                qml.RX(weights[..., idx], wires=q)
                qml.RY(weights[..., idx + 1], wires=q)
                qml.RZ(weights[..., idx + 2], wires=q)

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

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def apply_circuit_gates_only(inputs_np, weights_np, n_qubits: int, n_layers: int, entanglement: str):
    """
    Gate-application-only version (no measurement) for use with
    src/quantum/tomography.py's statevector_from_circuit(), which needs
    a function that ends with the circuit still "open" so qml.state()
    can be appended by the caller. Takes plain numpy arrays since
    tomography is a post-hoc analysis step, not part of the training
    graph — detaching here is fine (unlike in the generator's forward pass).
    """
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

    return lambda weights: _apply(weights if weights is not None else weights_np)


class _PQCFunction(torch.autograd.Function):
    """Differentiable wrapper around PQCRunner.

    forward : per-sample <Z_k> values (no gradient work at all when called
              under torch.no_grad(), e.g. synthetic-data generation).
    backward: ONE adjoint sweep per sample for the contracted gradient
              J^T g, instead of the full 20-observable Jacobian (~16x
              cheaper at 20 qubits; mathematically identical -- verified
              in tests/test_pqc_runner.py).
    """

    @staticmethod
    def forward(ctx, inputs, theta, runner):
        x = inputs.detach().cpu().numpy().astype(np.float64)
        th = theta.detach().cpu().numpy().astype(np.float64)
        out = runner.forward(x, th)
        ctx.runner = runner
        ctx.save_for_backward(inputs, theta)
        return torch.as_tensor(out, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_out):
        inputs, theta = ctx.saved_tensors
        need_x = bool(ctx.needs_input_grad[0])
        gx, gth = ctx.runner.vjp(
            inputs.detach().cpu().numpy().astype(np.float64),
            theta.detach().cpu().numpy().astype(np.float64),
            grad_out.detach().cpu().numpy().astype(np.float64),
            need_input_grad=need_x,
        )
        gx_t = torch.as_tensor(gx, dtype=inputs.dtype, device=inputs.device) if need_x else None
        return gx_t, torch.as_tensor(gth, dtype=theta.dtype, device=theta.device), None


class QLSTMGenerator(nn.Module):
    """
    Quantum LSTM-style generator. Same architectural intent as the
    original (parameterized quantum circuit -> classical projection
    layer), but now genuinely trainable end-to-end.
    """

    def __init__(self, n_qubits: int = 20, n_layers: int = 4, n_features: int = 32,
                 entanglement: str = "ring", noise_strength: float = 0.01,
                 quantum_device: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.entanglement = entanglement
        self.noise_strength = noise_strength
        self.quantum_device = quantum_device

        self.n_params = n_layers * n_qubits * 3
        self.theta = nn.Parameter(torch.randn(self.n_params) * 0.1)

        self.qnode = build_qlstm_qnode(n_qubits, n_layers, entanglement, quantum_device)
        # Fast path for lightning.* simulators (see _PQCFunction). Other
        # devices (default.qubit backprop, real QPUs) keep the original QNode path.
        self._runner = (PQCRunner(n_qubits, n_layers, entanglement, quantum_device)
                        if quantum_device.startswith("lightning") else None)
        self.output_layer = nn.Linear(n_qubits, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, n_features). Truncated/padded to n_qubits before encoding.
        Returns: (batch, n_features)
        """
        batch_size = z.shape[0]
        inputs = z[:, : self.n_qubits]
        if inputs.shape[1] < self.n_qubits:
            pad = torch.zeros(batch_size, self.n_qubits - inputs.shape[1], device=z.device, dtype=z.dtype)
            inputs = torch.cat([inputs, pad], dim=1)

        if self.training and self.noise_strength > 0:
            # Noise-as-a-feature (RQ3): additive Gaussian noise on the encoding,
            # kept inside the autograd graph (no detach) so its regularizing
            # effect on the trained parameters is real, not cosmetic.
            inputs = inputs + torch.randn_like(inputs) * self.noise_strength

        if self._runner is not None:
            stacked = _PQCFunction.apply(inputs, self.theta, self._runner).float()
        else:
            # PennyLane's torch interface supports batched execution when the
            # QNode's non-batch dims are consistent; broadcast over batch here.
            weights = self.theta.unsqueeze(0).expand(batch_size, -1)
            raw_outputs = self.qnode(inputs, weights)          # list of n_qubits tensors, each (batch,)
            stacked = torch.stack(raw_outputs, dim=-1).float()  # (batch, n_qubits)

        return self.output_layer(stacked)
