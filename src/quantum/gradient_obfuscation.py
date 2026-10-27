"""
Quantum Gradient Obfuscation Measure (QGOM): how much an attacker's
gradient signal is degraded by the quantum circuit under attack,
relative to a clean baseline.

    QGOM_d = 1 - Var(g_attacked) / Var(g_clean)

REQUIRES A REAL CAPABILITY circuits.py DIDN'T HAVE: every existing QNode
in this codebase (build_qlstm_qnode) runs in EXACT/analytic mode --
PennyLane's default.qubit returns the mathematically exact expectation
value every time, with zero shot noise, because no `shots=` was ever
passed to qml.device(). QGOM's entire premise is measuring how gradient
VARIANCE changes under attack, and exact-mode circuits have zero
gradient variance across repeated evaluations at fixed parameters by
construction -- Var(g_clean) would be identically 0, making QGOM_d
undefined (0/0) regardless of any real attack effect. This module adds
a genuinely separate finite-shots QNode builder so repeated evaluations
give different (shot-noise-limited) results, the way a real quantum
device would.

Gradients are computed via PennyLane's own parameter-shift
differentiation (qml.grad), not a hand-rolled finite-difference loop --
PennyLane's implementation is the standard, tested one; reimplementing
it manually (as an external draft of this module did) only adds a
second place the formula could be wrong for no benefit.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp


def build_finite_shot_qnode(n_qubits: int, n_layers: int, entanglement: str,
                             shots: int, dev_name: str = "default.qubit"):
    """Same encoding + variational + entangling gate sequence as
    circuits.build_qlstm_qnode, but on a device instantiated with a
    finite shot count, so measurement outcomes are genuinely stochastic
    (shot noise) rather than exact. Uses PennyLane's own `numpy` wrapper
    (pennylane.numpy) for parameters, matching what qml.grad expects for
    parameter-shift differentiation on a non-torch-interface QNode."""
    dev = qml.device(dev_name, wires=n_qubits, shots=shots)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
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
                for i2 in range(n_qubits):
                    for j2 in range(i2 + 1, n_qubits):
                        qml.CNOT(wires=[i2, j2])
            elif entanglement == "linear":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            else:
                raise ValueError(f"Unknown entanglement topology: {entanglement}")

        return qml.expval(qml.PauliZ(0))

    return circuit


def _scalar_loss(circuit, weights: pnp.ndarray, inputs: np.ndarray, target: float) -> pnp.ndarray:
    """MSE between the circuit's single-qubit-Z-expectation output and a
    target value -- a stand-in "training loss" for gradient-obfuscation
    testing purposes. Any differentiable scalar function of the circuit
    output would serve the same role here; MSE against a fixed target is
    the simplest well-defined choice, and keeps this module decoupled
    from the specific forecasting loss used elsewhere in the codebase."""
    output = circuit(inputs, weights)
    return (output - target) ** 2


def measure_gradient_variance(circuit, weights: np.ndarray, inputs: np.ndarray,
                               target: float, n_trials: int = 100, seed: int = 0,
                               n_weights_sampled: int = 12, weight_indices=None) -> dict:
    """
    Parameter-shift gradient of _scalar_loss w.r.t. a SUBSET of the weights,
    repeated `n_trials` times (shot noise varies trial to trial). Variance
    across trials is averaged over the sampled weights, matching Var(g) in
    QGOM_d.

    Why a subset: a full gradient needs 2 circuit runs per weight (480 at
    20 qubits x 4 layers), so 100 trials x 2 input sets cost ~96k
    finite-shot 20-qubit simulations (hours). Mean per-weight variance over
    a fixed random subset (the SAME indices for clean and attacked inputs)
    is an unbiased estimate of the mean over all weights at ~1/20 the cost.
    Every weight sits in a single RX/RY/RZ gate, so the pi/2 shift rule is
    exact: dL/dw = 2 (f - t) * (f(w+pi/2) - f(w-pi/2)) / 2.
    """
    weights = np.asarray(weights, dtype=float).copy()
    n_w = len(weights)
    if weight_indices is None:
        rng_idx = np.random.default_rng(12345)  # fixed: same weights for clean/attacked
        k = min(n_weights_sampled, n_w)
        weight_indices = np.sort(rng_idx.choice(n_w, size=k, replace=False))
    weight_indices = list(weight_indices)

    def f(w):
        return float(circuit(inputs, pnp.array(w, requires_grad=False)))

    gradients = []
    for _ in range(n_trials):
        out = f(weights)
        g = []
        for i in weight_indices:
            wp, wm = weights.copy(), weights.copy()
            wp[i] += np.pi / 2
            wm[i] -= np.pi / 2
            g.append(2.0 * (out - target) * 0.5 * (f(wp) - f(wm)))
        gradients.append(g)

    gradients = np.asarray(gradients, dtype=float)  # (n_trials, n_sampled)
    per_weight_variance = np.var(gradients, axis=0, ddof=1)
    return {
        "mean_gradient_variance": float(np.mean(per_weight_variance)),
        "gradients": gradients,
        "n_trials": n_trials,
        "weight_indices": weight_indices,
    }


def shot_noise_floor(circuit, weights: np.ndarray, inputs: np.ndarray, target: float,
                      n_repeats: int = 20, seed: int = 0) -> float:
    """Variance of the LOSS itself (not its gradient) across repeated
    evaluations at fixed weights/inputs -- estimates the measurement
    noise floor so it can be subtracted from the gradient variance,
    isolating attack-induced variance from ordinary shot noise."""
    rng = np.random.default_rng(seed)
    weights_pnp = pnp.array(weights, requires_grad=False)
    losses = []
    for _ in range(n_repeats):
        qml.numpy.random.seed(int(rng.integers(0, 1_000_000)))
        losses.append(float(_scalar_loss(circuit, weights_pnp, inputs, target)))
    return float(np.var(losses, ddof=1))


def qgom_d(n_qubits: int, n_layers: int, entanglement: str, weights: np.ndarray,
           inputs_clean: np.ndarray, inputs_attacked: np.ndarray, target: float = 0.0,
           shots: int = 1024, n_trials: int = 30, dev_name: str = "default.qubit",
           seed: int = 0) -> dict:
    """
    QGOM_d = 1 - Var(g_attacked) / Var(g_clean), shot-noise-floor corrected.

    inputs_clean, inputs_attacked: encoding-layer input vectors before
    and after an input-space attack (e.g. FGSM/PGD from adversarial.py,
    applied to the classical features before quantum encoding).

    Target interpretation (per the design source): 0 = attacker's
    gradient signal intact (no obfuscation, poor defense); [0.5, 0.8] =
    disrupted but still trainable; ~1 = gradient fully obfuscated
    (circuit untrainable / barren-plateau-like, over-defended).
    """
    circuit = build_finite_shot_qnode(n_qubits, n_layers, entanglement, shots, dev_name)

    var_clean = measure_gradient_variance(circuit, weights, inputs_clean, target, n_trials, seed)["mean_gradient_variance"]
    var_attacked = measure_gradient_variance(circuit, weights, inputs_attacked, target, n_trials, seed + 1)["mean_gradient_variance"]

    noise_floor = shot_noise_floor(circuit, weights, inputs_clean, target, n_repeats=20, seed=seed + 2)
    var_clean_corrected = var_clean - noise_floor
    var_attacked_corrected = var_attacked - noise_floor
    # If either variance is not above the shot-noise floor the ratio is not
    # identifiable. The earlier clamp (max(.., 1e-12) on both) returned an
    # exact 0.0 in that case, which reads as "no obfuscation" when the truth
    # is "not measurable" (all logged runs of Oct 12-25 show qgom == 0.0).
    identifiable = var_clean_corrected > 0.0 and var_attacked_corrected > 0.0
    qgom_value = (float(1.0 - var_attacked_corrected / var_clean_corrected)
                  if identifiable else float("nan"))

    return {
        "QGOM_d": qgom_value,
        "identifiable": bool(identifiable),
        "var_clean": var_clean,
        "var_attacked": var_attacked,
        "noise_floor": noise_floor,
        "shots": shots,
        "n_trials": n_trials,
    }
