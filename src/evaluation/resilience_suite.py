"""
Per-quantum-model resilience suite: QSFR, EER (attacks/quantum_attacks.py),
QGOM (quantum/gradient_obfuscation.py), M1_EFI (quantum/encoding_fidelity.py),
and NLCS (metrics/resilience.py, rescoped to a reduced subsystem -- see
that module's docstring for why). None of these four modules were
called from anywhere in the actual pipeline before this file -- only
from their own tests.

CTR is deliberately NOT computed here: it needs a genuine decoherence-
channel time-series simulator (e.g. PennyLane's default.mixed device
with AmplitudeDamping/PhaseDamping channels) that doesn't exist anywhere
in this codebase yet -- see metrics/resilience.py's ctr() docstring.
That's a separate, not-yet-built piece of work (tracked in TODO.md), not
something reachable by calling existing pieces together. CQRS (which
combines all five, and gracefully renormalizes when one is missing) is
computed in scripts/run_hypothesis_tests.py.

Only meaningful for models with an actual trained quantum circuit
(flat weight vector, n_qubits, n_layers, entanglement topology) --
classical baselines have none of this and must not call
run_resilience_suite.
"""
from __future__ import annotations

import numpy as np
import torch

from ..attacks.adversarial import fgsm_perturbation
from ..attacks.quantum_attacks import (
    quantum_noise_injection_attack,
    entanglement_disruption_attack,
    _clean_gate_sequence,
)
from ..quantum.tomography import statevector_from_circuit, reduced_density_matrix
from ..quantum.gradient_obfuscation import qgom_d
from ..quantum.encoding_fidelity import compute_m1_efi
from ..quantum.decoherence import decoherence_rho_series
from ..metrics.resilience import nlcs, ctr, check_ctr_tractable, MAX_QUBITS_FOR_DENSITY_MATRIX


def run_resilience_suite(forward_fn, weights_np: np.ndarray, X_single: np.ndarray,
                          y_single: float, X_batch: np.ndarray, n_qubits: int, n_layers: int,
                          entanglement: str, dev_name: str = "default.qubit",
                          seed: int = 0) -> dict:
    """
    forward_fn: the model's classical, differentiable forecast callable
    (e.g. self._forecast_model) -- used only to produce an FGSM-attacked
    classical input for QGOM's clean-vs-attacked comparison, reusing
    adversarial.py's existing attack rather than a new one. This is
    exactly the "input-space attack ... applied to the classical
    features before quantum encoding" QGOM's own docstring calls for.
    weights_np: the trained circuit's flat parameter array (e.g.
    self.theta.detach().numpy()).
    X_single: one row of real input features (length >= n_qubits) for
    QSFR/EER/QGOM/NLCS, which are single-sample APIs.
    X_batch: a modest batch of real rows (e.g. 50) for M1_EFI, which
    characterizes the encoding map's behavior over a distribution, not
    one point.
    """
    result: dict = {}
    inputs_np = np.asarray(X_single[:n_qubits], dtype=np.float64)

    try:
        qsfr = quantum_noise_injection_attack(inputs_np, weights_np, n_qubits, n_layers,
                                               entanglement, dev_name=dev_name, seed=seed)
        result["qsfr"] = qsfr["qsfr"]
    except Exception as exc:
        result["qsfr"] = float("nan")
        result["qsfr_error"] = str(exc)

    try:
        eer = entanglement_disruption_attack(inputs_np, weights_np, n_qubits, n_layers,
                                              entanglement, dev_name=dev_name, seed=seed)
        result["eer"] = eer["eer"]
    except Exception as exc:
        result["eer"] = float("nan")
        result["eer_error"] = str(exc)

    # NLCS: recomputes the same clean/attacked reduced states
    # quantum_noise_injection_attack derives internally (that function
    # returns only the scalar QSFR, not the matrices) via the same
    # shared, tractable-subsystem helper -- not a claim of a different
    # attack, just re-deriving the intermediate density matrices.
    try:
        cut = n_qubits // 2
        subsystem = list(range(cut))
        clean_fn = _clean_gate_sequence(inputs_np, n_qubits, n_layers, entanglement)
        state_clean = statevector_from_circuit(clean_fn, weights_np, n_qubits, dev_name)
        rho_clean = reduced_density_matrix(state_clean, n_qubits, subsystem)
        rng = np.random.default_rng(seed)
        weights_attacked = weights_np + rng.normal(0, 0.3, size=weights_np.shape)
        state_attacked = statevector_from_circuit(clean_fn, weights_attacked, n_qubits, dev_name)
        rho_attacked = reduced_density_matrix(state_attacked, n_qubits, subsystem)
        result["nlcs"] = nlcs(rho_clean, rho_attacked)
    except Exception as exc:
        result["nlcs"] = float("nan")
        result["nlcs_error"] = str(exc)

    # CTR -- now computable (src/quantum/decoherence.py), but still capped
    # at MAX_QUBITS_FOR_DENSITY_MATRIX (12): a dense density matrix for this
    # study's primary 20-qubit configuration would need ~1.1 TB, regardless
    # of how the decoherence series is generated -- that ceiling was never
    # the missing piece, the series-generator was, and that's fixed now.
    if n_qubits <= MAX_QUBITS_FOR_DENSITY_MATRIX:
        try:
            # decoherence_rho_series always uses PennyLane's default.mixed
            # device (the only device here that supports noise channels) --
            # unlike every other function in this module, it has no
            # dev_name parameter to pass through.
            clean_series = decoherence_rho_series(
                inputs_np, weights_np, n_qubits, n_layers, entanglement,
                extra_depolarizing=0.0, seed=seed,
            )
            attacked_series = decoherence_rho_series(
                inputs_np, weights_np, n_qubits, n_layers, entanglement,
                extra_depolarizing=0.15, seed=seed,
            )
            result["ctr"] = ctr(clean_series, attacked_series, n_qubits=n_qubits)
        except Exception as exc:
            result["ctr"] = float("nan")
            result["ctr_error"] = str(exc)
    else:
        result["ctr"] = float("nan")
        result["ctr_note"] = (
            f"not computed: n_qubits={n_qubits} exceeds the "
            f"{MAX_QUBITS_FOR_DENSITY_MATRIX}-qubit dense-density-matrix ceiling "
            f"(this study's primary 20-qubit configuration needs ~1.1 TB regardless "
            f"of how the decoherence series is generated -- see check_ctr_tractable)"
        )

    try:
        X_t = torch.as_tensor(X_single.reshape(1, -1), dtype=torch.float32)
        y_t = torch.as_tensor([float(y_single)], dtype=torch.float32)
        X_attacked_t = fgsm_perturbation(forward_fn, X_t, y_t, epsilon=0.1)
        qgom = qgom_d(
            n_qubits, n_layers, entanglement, weights_np,
            inputs_clean=X_single[:n_qubits],
            inputs_attacked=X_attacked_t.detach().numpy().flatten()[:n_qubits],
            target=float(y_single), dev_name=dev_name, seed=seed,
        )
        result["qgom"] = qgom["QGOM_d"]
    except Exception as exc:
        result["qgom"] = float("nan")
        result["qgom_error"] = str(exc)

    try:
        efi = compute_m1_efi(np.asarray(X_batch, dtype=np.float64), weights_np, n_qubits,
                              n_layers, entanglement, dev_name=dev_name, seed=seed)
        result["m1_efi"] = efi["M1_EFI"]
    except Exception as exc:
        result["m1_efi"] = float("nan")
        result["m1_efi_error"] = str(exc)

    return result
