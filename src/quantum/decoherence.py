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
from ..metrics.resilience import check_ctr_tractable, MAX_QUBITS_FOR_DENSITY_MATRIX
from .tomography import reduced_density_matrix


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


# ======================================================================
# TRAJECTORY (Monte Carlo wavefunction) VERSION -- scales past the
# ~16 TB dense-density-matrix ceiling of decoherence_rho_series above.
# ======================================================================
"""
decoherence_rho_series() above needs a full (2**n x 2**n) density
matrix throughout -- 16 TB at this study's primary 20-qubit
configuration, not a missing feature but a hard physical ceiling on
ANY implementation that represents the full mixed state directly.

decoherence_rho_series_trajectory() avoids that entirely by never
representing a mixed state at all. It's the standard technique for
simulating open quantum systems at scale (the "quantum trajectory" or
"Monte Carlo wavefunction" method, widely used in the open-quantum-
systems literature): a quantum channel rho -> sum_i K_i rho K_i^dagger
is mathematically IDENTICAL, in expectation, to repeatedly (a) picking
Kraus operator K_i stochastically, with probability equal to the
post-application state's squared norm, and (b) applying and
renormalizing it to a PURE state -- then averaging the resulting pure
states' (reduced) density matrices over many independent trajectories.
Each trajectory is a plain statevector -- O(2**n) memory, 16 MB at
n=20, the same order of magnitude every other quantum computation in
this codebase already runs at, not 16 TB.

Trade-off: statistical, not exact. More trajectories -> lower variance,
at proportionally higher cost. Defaults (n_trajectories=200) are a
starting point, not a validated-sufficient sample size for every use --
see the convergence note on n_trajectories below.
"""


def _amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    return [
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex),
    ]


def _phase_damping_kraus(lam: float) -> list[np.ndarray]:
    return [
        np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
        np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex),
    ]


def _depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Must match qml.DepolarizingChannel(p)'s own convention EXACTLY,
    since decoherence_rho_series() (the exact, density-matrix version
    this trajectory method is cross-validated against) uses that
    PennyLane operation directly: K0 = sqrt(1-p)*I, K1..3 = sqrt(p/3)*{X,Y,Z}
    -- i.e. p is literally "probability an X/Y/Z error happens, p/3 each."
    A first version of this function used sqrt(1-3p/4)/sqrt(p/4) instead
    (a different, non-equivalent depolarizing-channel parameterization),
    which matched PennyLane's own Kraus matrices for every OTHER channel
    here but not this one -- caught by cross-validating against
    decoherence_rho_series() at a small qubit count, where the clean
    (no-depolarizing) condition matched to within statistical noise but
    the attacked (depolarizing-involved) condition did not, and did not
    improve with more trajectories (the signature of a formula bug, not
    noise) -- see tests/test_decoherence_trajectory.py."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [np.sqrt(max(1 - p, 0)) * I, np.sqrt(p / 3) * X,
            np.sqrt(p / 3) * Y, np.sqrt(p / 3) * Z]


def _apply_local_operator(psi: np.ndarray, op: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    """Applies a single-qubit (2x2, not necessarily unitary) operator to
    `wire` of an n-qubit statevector. Returns the UN-normalized result --
    ||result||**2 is the probability of this specific Kraus outcome,
    needed by the caller before renormalizing."""
    psi_t = psi.reshape([2] * n_qubits)
    psi_t = np.moveaxis(psi_t, wire, 0)
    rest_shape = psi_t.shape[1:]
    psi_t = op @ psi_t.reshape(2, -1)
    psi_t = psi_t.reshape((2,) + rest_shape)
    return np.moveaxis(psi_t, 0, wire).reshape(-1)


def _sample_and_apply_channel(psi: np.ndarray, kraus_ops: list[np.ndarray], wire: int,
                               n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    """Stochastically applies ONE Kraus operator from kraus_ops to `wire`,
    selected with probability equal to the resulting squared norm, then
    renormalizes. See this section's module-level docstring for why this
    exactly reproduces the channel's average behavior over many calls."""
    candidates = [_apply_local_operator(psi, k, wire, n_qubits) for k in kraus_ops]
    probs = np.array([np.vdot(c, c).real for c in candidates])
    total = probs.sum()
    probs = probs / total if total > 0 else np.ones(len(kraus_ops)) / len(kraus_ops)
    choice = rng.choice(len(kraus_ops), p=probs)
    result = candidates[choice]
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-15 else result


def decoherence_rho_series_trajectory(inputs_np: np.ndarray, weights_np: np.ndarray, n_qubits: int,
                                       n_layers: int, entanglement: str, n_steps: int = 10,
                                       dt: float = 0.1, t1: float = 5.0, t2_dephasing: float = 5.0,
                                       extra_depolarizing: float = 0.0, subsystem: list[int] = None,
                                       n_trajectories: int = 200, seed: int = 0) -> list[np.ndarray]:
    """
    Trajectory-based equivalent of decoherence_rho_series() -- same
    inputs, same return shape (a list of n_steps reduced density
    matrices on `subsystem`), scales to any n_qubits this codebase's
    pure-state circuits already handle (no MAX_QUBITS_FOR_DENSITY_MATRIX
    ceiling, since no dense density matrix is ever built).

    n_trajectories controls the statistical precision of the result (law
    of large numbers, standard error ~ 1/sqrt(n_trajectories)) -- NOT a
    simulation-fidelity knob the way n_qubits/n_layers are elsewhere in
    this codebase. Validated in tests/test_decoherence_trajectory.py
    against: (1) the exact analytical single-qubit amplitude-damping
    decay curve, and (2) the dense decoherence_rho_series()'s CTR value
    at a small, tractable qubit count both methods can run at.
    """
    if subsystem is None:
        subsystem = list(range(max(n_qubits // 2, 1)))

    gamma_amp = 1.0 - np.exp(-dt / t1)
    gamma_phase = 1.0 - np.exp(-dt / t2_dephasing)
    amp_kraus = _amplitude_damping_kraus(gamma_amp)
    phase_kraus = _phase_damping_kraus(gamma_phase)
    dep_kraus = _depolarizing_kraus(extra_depolarizing) if extra_depolarizing > 0 else None

    clean_fn = _clean_gate_sequence(inputs_np, n_qubits, n_layers, entanglement)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def prep():
        clean_fn(weights_np)
        return qml.state()

    psi0 = np.asarray(prep())
    dim_keep = 2 ** len(subsystem)
    accum = [np.zeros((dim_keep, dim_keep), dtype=complex) for _ in range(n_steps)]

    rng = np.random.default_rng(seed)
    for _ in range(n_trajectories):
        psi = psi0.copy()
        for step in range(n_steps):
            for w in range(n_qubits):
                psi = _sample_and_apply_channel(psi, amp_kraus, w, n_qubits, rng)
                psi = _sample_and_apply_channel(psi, phase_kraus, w, n_qubits, rng)
                if dep_kraus is not None:
                    psi = _sample_and_apply_channel(psi, dep_kraus, w, n_qubits, rng)
            accum[step] += reduced_density_matrix(psi, n_qubits, subsystem)

    return [a / n_trajectories for a in accum]
