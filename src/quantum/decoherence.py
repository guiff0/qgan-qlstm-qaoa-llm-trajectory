"""
Builds the decoherence-channel time-series `metrics/resilience.py`'s
ctr() needs but this codebase never generated anywhere -- see that
function's own docstring: "Generating that series requires a
decoherence-channel simulator ... that this codebase does not
currently build anywhere."

Uses PennyLane's `default.mixed` device -- the one device anywhere in
this codebase that represents genuine mixed states and supports noise
channels (AmplitudeDamping/PhaseDamping/DepolarizingChannel).
default.qubit/lightning.qubit, used everywhere else here, are PURE-
STATE simulators and cannot represent decoherence at all. This is the
only module that uses default.mixed.

SCOPE NOTE: T1 (amplitude damping) and T2 (phase/dephasing) times below
are simulation PARAMETERS, not measured hardware characteristics --
this project has no real quantum hardware. They're chosen to produce a
visible, fittable decay over a modest number of steps (n_steps=10
default), not calibrated against any physical device. Report them
alongside any CTR number this produces.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

from ..attacks.quantum_attacks import _clean_gate_sequence
from ..metrics.resilience import check_ctr_tractable


def partial_trace_density_matrix(rho: np.ndarray, n_qubits: int, keep: list[int]) -> np.ndarray:
    """
    Partial trace of a general (mixed) density matrix over the
    complement of `keep`. quantum/tomography.py's reduced_density_matrix
    only handles PURE states (it computes |psi><psi| and traces that) --
    this handles a genuine mixed rho, which is what a decoherence
    channel's output actually is.

    rho: (2**n_qubits, 2**n_qubits) array. keep: qubit indices to retain,
    in their original order.
    """
    trace_out = [q for q in range(n_qubits) if q not in keep]
    rho_t = rho.reshape([2] * (2 * n_qubits))  # axes: row_0..row_{n-1}, col_0..col_{n-1}

    n_remaining = n_qubits
    row_axis_of = list(range(n_qubits))  # row_axis_of[original_qubit] -> current row axis position
    for q in sorted(trace_out, reverse=True):
        row_axis = row_axis_of[q]
        col_axis = row_axis + n_remaining
        rho_t = np.trace(rho_t, axis1=row_axis, axis2=col_axis)
        n_remaining -= 1
        for k in range(n_qubits):
            if row_axis_of[k] is not None and row_axis_of[k] > row_axis:
                row_axis_of[k] -= 1
        row_axis_of[q] = None

    dim_keep = 2 ** len(keep)
    return rho_t.reshape(dim_keep, dim_keep)


def decoherence_rho_series(inputs_np: np.ndarray, weights_np: np.ndarray, n_qubits: int,
                            n_layers: int, entanglement: str, n_steps: int = 10,
                            dt: float = 0.1, t1: float = 5.0, t2_dephasing: float = 5.0,
                            extra_depolarizing: float = 0.0, subsystem: list[int] = None,
                            seed: int = 0) -> list[np.ndarray]:
    """
    Prepares the trained circuit's state, then applies n_steps of
    AmplitudeDamping (T1) + PhaseDamping (T2) per qubit per step,
    returning the reduced density matrix (on `subsystem`, default the
    first half of qubits -- the same tractable-subsystem convention
    used throughout this codebase's density-matrix work) after each
    step. check_ctr_tractable() is called first with the FULL n_qubits
    (the state() call below needs the full system, even though only a
    reduced subsystem is returned) -- same ceiling ctr() itself enforces.

    extra_depolarizing > 0 models a degraded/"attacked" channel (e.g.
    environmental noise worse than the clean baseline) -- call this
    with extra_depolarizing=0 for rho_clean_series and > 0 for
    rho_attacked_series; ctr() compares the two resulting T2 fits.
    """
    check_ctr_tractable(n_qubits)
    if subsystem is None:
        subsystem = list(range(max(n_qubits // 2, 1)))

    dev = qml.device("default.mixed", wires=n_qubits, seed=seed)
    clean_fn = _clean_gate_sequence(inputs_np, n_qubits, n_layers, entanglement)

    gamma_amp = 1.0 - np.exp(-dt / t1)
    gamma_phase = 1.0 - np.exp(-dt / t2_dephasing)

    series = []
    for step in range(1, n_steps + 1):
        @qml.qnode(dev)
        def circuit(step=step):
            clean_fn(weights_np)
            for _ in range(step):
                for w in range(n_qubits):
                    qml.AmplitudeDamping(gamma_amp, wires=w)
                    qml.PhaseDamping(gamma_phase, wires=w)
                    if extra_depolarizing > 0:
                        qml.DepolarizingChannel(extra_depolarizing, wires=w)
            return qml.density_matrix(wires=range(n_qubits))

        full_rho = np.asarray(circuit())
        series.append(partial_trace_density_matrix(full_rho, n_qubits, subsystem))
    return series
