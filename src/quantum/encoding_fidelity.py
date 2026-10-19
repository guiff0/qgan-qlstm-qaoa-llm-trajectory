"""
M1 = Encoding Fidelity Index (EFI): a genuine, non-fabricated
implementation of the four sub-metrics from the confirmed operational
definition, computed on the circuit actually implemented in this
codebase (see encoding.py's module docstring for the amplitude-vs-angle
encoding gap this deliberately does NOT paper over).

Every sub-metric below is computed from a real PennyLane statevector,
not a constant, not a comparison of data to itself-plus-noise, and not
a hardcoded lookup keyed on a categorical label -- the three failure
patterns found across the uploaded repository's equivalent file. Each
function's docstring states exactly what it computes and what
simplification (if any) was necessary to keep it tractable at 20
qubits, rather than silently computing something narrower than its name
implies.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

from .encoding import amplitude_encode
from ..evaluation.mediation import mediation_analysis  # noqa: F401 (re-export convenience for callers)


# ---------------------------------------------------------------------------
# Shared circuit execution (duplicated from circuits.py / quantum_attacks.py
# by the same deliberate choice documented in quantum_attacks.py: this file
# needs the statevector at a specific point, which the shared QNode helpers
# don't expose a hook for. Keep in sync with circuits.build_qlstm_qnode if
# the base architecture changes.)
# ---------------------------------------------------------------------------

def _run_circuit_statevector(inputs_np: np.ndarray, weights_np: np.ndarray, n_qubits: int,
                              n_layers: int, entanglement: str, dev_name: str = "default.qubit") -> np.ndarray:
    dev = qml.device(dev_name, wires=n_qubits)

    def _apply_entangling():
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

    @qml.qnode(dev, interface=None)
    def _circuit(weights):
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
            _apply_entangling()
        return qml.state()

    return np.asarray(_circuit(weights_np))


# ---------------------------------------------------------------------------
# Sub-metric 1: State Preparation Fidelity (SPF)
# ---------------------------------------------------------------------------

def state_preparation_fidelity(X: np.ndarray, weights: np.ndarray, n_qubits: int,
                                n_layers: int, entanglement: str,
                                dev_name: str = "default.qubit") -> float:
    """
    SPF = mean over samples of |<psi_ideal | psi_circuit>|^2, where
    psi_ideal is the deterministic amplitude-encoding reference
    (encoding.amplitude_encode) and psi_circuit is this codebase's
    ACTUAL circuit (angle encoding + trained variational layers),
    executed for real via PennyLane.

    This is a genuine fidelity between two different pure states, not
    a trivial self-comparison: because the implemented circuit encodes
    via angle rotations, not amplitude embedding, psi_circuit will
    generally NOT equal psi_ideal, and SPF measures exactly how far
    apart they are. A circuit that happened to implement true amplitude
    encoding would score SPF = 1.0 by construction; this one does not,
    and that is real information, not a bug.
    """
    ideal_states = amplitude_encode(X, n_qubits)
    fidelities = []
    for i in range(X.shape[0]):
        psi_circuit = _run_circuit_statevector(X[i], weights, n_qubits, n_layers, entanglement, dev_name)
        overlap = np.vdot(ideal_states[i], psi_circuit)
        fidelities.append(float(np.abs(overlap) ** 2))
    return float(np.mean(fidelities))


# ---------------------------------------------------------------------------
# Sub-metric 2: Information Preservation (IP)
# ---------------------------------------------------------------------------

def information_preservation(X: np.ndarray, weights: np.ndarray, n_qubits: int,
                              n_layers: int, entanglement: str,
                              dev_name: str = "default.qubit") -> float:
    """
    IP measures how well each qubit's Z-basis measurement probability
    tracks its corresponding (normalized) input feature.

    SIMPLIFICATION, STATED EXPLICITLY: the operational definition frames
    IP as 1 - KL(classical distribution || measured distribution). A
    literal reading would compare the classical feature distribution to
    the FULL 2^n_qubits-outcome measurement distribution -- intractable
    to estimate by sampling at n_qubits=20 (over a million possible
    outcomes; no realistic sample size resolves that histogram). Instead,
    IP here is computed per-qubit: each qubit's real P(measure |1>),
    computed exactly from the statevector (no sampling noise), compared
    to that qubit's normalized input feature value, and averaged. This
    is a genuine measurement-based quantity, not a comparison of the
    data to itself -- but it is a per-qubit marginal comparison, not the
    full joint-distribution KL divergence the name might imply.
    """
    n = min(n_qubits, X.shape[1])
    feature_min = X[:, :n].min(axis=0)
    feature_max = X[:, :n].max(axis=0)
    feature_range = np.where(feature_max > feature_min, feature_max - feature_min, 1.0)
    normalized_features = (X[:, :n] - feature_min) / feature_range  # -> [0, 1] per feature

    diffs = []
    for i in range(X.shape[0]):
        psi = _run_circuit_statevector(X[i], weights, n_qubits, n_layers, entanglement, dev_name)
        probs = np.abs(psi) ** 2
        dim = len(probs)
        p1_per_qubit = np.zeros(n_qubits)
        for q in range(n_qubits):
            mask = (np.arange(dim) >> q) & 1
            p1_per_qubit[q] = probs[mask == 1].sum()
        diffs.append(np.abs(normalized_features[i] - p1_per_qubit[:n]).mean())

    return float(1.0 - np.mean(diffs))


# ---------------------------------------------------------------------------
# Sub-metric 3: Expressibility Deficit (ED)
# ---------------------------------------------------------------------------

def _haar_fidelity_pdf(F: np.ndarray, dim: int) -> np.ndarray:
    """Analytic Haar-random pairwise-fidelity density for a Hilbert
    space of dimension `dim` (Sim, Johnson & Aspuru-Guzik, 2019)."""
    return (dim - 1) * (1 - F) ** (dim - 2)


def expressibility_deficit(n_qubits: int, n_layers: int, entanglement: str,
                            n_samples: int = 200, n_bins: int = 20,
                            dev_name: str = "default.qubit", seed: int = 0) -> float:
    """
    Real expressibility measure: sample n_samples random parameter
    vectors, compute pairwise fidelities between the resulting circuit
    output states, histogram them, and compute the KL divergence
    against the analytic Haar-random fidelity distribution for this
    Hilbert space dimension. Lower ED = the circuit's outputs are
    harder to distinguish from Haar-random states = more expressive.

    COST NOTE: this requires n_samples circuit executions (each a full
    2^n_qubits statevector) plus O(n_samples^2) fidelity evaluations.
    At n_qubits=20 this is the most expensive of the four sub-metrics;
    reduce n_samples for a faster, noisier estimate, or run this
    sub-metric on a subsample of the full test set rather than every
    row, unlike SPF/IP/EI above.
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    n_params = n_layers * n_qubits * 3

    states = []
    for _ in range(n_samples):
        inputs = rng.uniform(-np.pi, np.pi, size=n_qubits)
        weights = rng.uniform(-np.pi, np.pi, size=n_params)
        states.append(_run_circuit_statevector(inputs, weights, n_qubits, n_layers, entanglement, dev_name))

    fidelities = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fidelities.append(np.abs(np.vdot(states[i], states[j])) ** 2)
    fidelities = np.array(fidelities)

    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=False)
    empirical = hist / hist.sum() if hist.sum() > 0 else np.ones(n_bins) / n_bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # (dim-1)(1-F)^(dim-2) underflows to 0 for dim = 2^20, giving 0/0 = NaN.
    # Normalise in log space instead.
    log_pdf = np.log(dim - 1) + (dim - 2) * np.log1p(-np.clip(bin_centers, 0.0, 1.0 - 1e-12))
    log_pdf = log_pdf - log_pdf.max()
    haar_probs = np.exp(log_pdf)
    haar_probs = haar_probs / haar_probs.sum()

    empirical = empirical + 1e-12
    haar_probs = haar_probs + 1e-12
    kl = float(np.sum(empirical * np.log(empirical / haar_probs)))
    return kl


# ---------------------------------------------------------------------------
# Sub-metric 4: Encoding Injectivity (EI)
# ---------------------------------------------------------------------------

def encoding_injectivity(X: np.ndarray, weights: np.ndarray, n_qubits: int, n_layers: int,
                          entanglement: str, n_pairs: int = 200, fidelity_threshold: float = 0.9,
                          dev_name: str = "default.qubit", seed: int = 0) -> float:
    """
    EI = 1 - (fraction of sampled DISTINCT-input pairs whose encoded
    quantum states are still nearly indistinguishable, i.e. fidelity
    above fidelity_threshold).

    Computed on the actual encoded statevectors (not raw classical
    Euclidean distance), so a "collision" here specifically means two
    different inputs the circuit maps to states that are hard to tell
    apart on a real quantum measurement.
    """
    n = X.shape[0]
    if n < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    idx_pairs = rng.integers(0, n, size=(n_pairs, 2))

    states_cache: dict[int, np.ndarray] = {}
    def _get_state(i):
        if i not in states_cache:
            states_cache[i] = _run_circuit_statevector(X[i], weights, n_qubits, n_layers, entanglement, dev_name)
        return states_cache[i]

    collisions = 0
    valid_pairs = 0
    for i, j in idx_pairs:
        if i == j:
            continue
        valid_pairs += 1
        fidelity = np.abs(np.vdot(_get_state(i), _get_state(j))) ** 2
        if fidelity > fidelity_threshold:
            collisions += 1

    return float(1.0 - collisions / valid_pairs) if valid_pairs > 0 else 1.0


# ---------------------------------------------------------------------------
# Composite M1
# ---------------------------------------------------------------------------

def compute_m1_efi(X: np.ndarray, weights: np.ndarray, n_qubits: int, n_layers: int,
                    entanglement: str, dev_name: str = "default.qubit",
                    ed_n_samples: int = 200, seed: int = 0) -> dict:
    """
    M1 = mean(SPF, IP, 1-ED, EI), all four computed for real against
    the circuit actually implemented in circuits.py -- measured on the
    encoding circuit's own output states, before any GAN/discriminator
    training, per the confirmed operational definition (M1 is a
    property of the encoding map, not of the generated data).
    """
    spf = state_preparation_fidelity(X, weights, n_qubits, n_layers, entanglement, dev_name)
    ip = information_preservation(X, weights, n_qubits, n_layers, entanglement, dev_name)
    ed = expressibility_deficit(n_qubits, n_layers, entanglement, n_samples=ed_n_samples,
                                 dev_name=dev_name, seed=seed)
    ei = encoding_injectivity(X, weights, n_qubits, n_layers, entanglement, dev_name=dev_name, seed=seed)

    m1 = float(np.mean([spf, ip, 1.0 - ed, ei]))
    return {"SPF": spf, "IP": ip, "ED": ed, "EI": ei, "M1_EFI": m1, "n_qubits": n_qubits}
