"""
Resilience metrics: QAR, CQRS, NLCS, and CTR -- the remaining four of
Table 6's six metrics not already covered by quantum_attacks.py (QSFR,
EER) and gradient_obfuscation.py (QGOM).

TWO REAL PROBLEMS IN THE SOURCE DESIGN, FIXED HERE RATHER THAN PAPERED OVER:

1. CTR (Coherence Time Retention) requires a genuine density-matrix
   time-evolution under a decoherence channel -- T2 has no meaning for
   a pure-state, noiseless simulator. A density matrix at n qubits is
   4^n entries, not 2^n: at this study's primary 20-qubit configuration,
   that is 4^20 ~= 1.1 TRILLION entries -- not slow, not expensive,
   ACTUALLY INTACTABLE to hold in memory as a dense array (roughly 17.6
   TB at complex128). This was already flagged in chat and is not solved
   by better code; it is a hardware/scale constraint. ctr() below
   enforces this with a hard, explicit check rather than letting a
   20-qubit call silently hang or OOM: it raises a clear error naming
   the actual memory requirement, and only executes for qubit counts
   where a dense density matrix is realistically tractable (<=12
   qubits, ~4.3 GB at complex128 -- still large, but a documented,
   explicit ceiling rather than a surprise crash).

2. NLCS (Non-Local Correlation Stability), as specified in the source
   document, sums |rho_ij| over the FULL system's off-diagonal elements
   at Hamming distance >= 2, with a Monte Carlo SAMPLE of index pairs
   proposed as the fix for tractability. That sampling scheme reduces
   the number of index-pair evaluations, but does NOT reduce the size
   of `rho` itself -- indexing rho[i, j] still requires the full (2^20,
   2^20) matrix to already exist in memory, which is the exact same
   4^n wall as CTR. Sampling a matrix you cannot construct doesn't help.
   The fix here is different: NLCS is computed on the REDUCED density
   matrix over a fixed-size subsystem (the same partial-trace subsystem
   size used by EER, quantum_attacks.entanglement_disruption_attack),
   not the full system. This is a genuine rescoping (non-local
   correlations within a tractable subsystem, not an approximation of
   the full-system quantity) and is stated as such -- it is not the same
   claim as "non-local correlation across the entire 20-qubit state."
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# QAR -- Quantum Advantage Retention (pure arithmetic on real ASR values;
# no tomography, no memory constraints, no simplification needed)
# ---------------------------------------------------------------------------

def qar(asr_classical_clean: float, asr_quantum_clean: float,
        asr_classical_attacked: float, asr_quantum_attacked: float) -> float:
    """
    QAR = (ASR_classical - ASR_quantum)_attacked / (ASR_classical - ASR_quantum)_clean

    All four inputs are real ASR values already produced by this
    codebase's existing, tested attack pipeline (src/attacks/adversarial.py's
    compute_attack_success_rate, run under clean and attacked conditions
    for both the classical and quantum baselines) -- nothing here is
    computed from tomography or requires new measurement infrastructure.

    QAR = 1: the full clean-condition quantum advantage persists under
    attack. QAR = 0: the advantage is fully erased. QAR < 0: the
    quantum model's relative advantage REVERSED under attack (the
    classical model became the more resilient one) -- report this
    explicitly rather than clipping it to 0, since a negative QAR is a
    real, reportable finding, not an error.
    """
    numerator = asr_classical_attacked - asr_quantum_attacked
    denominator = asr_classical_clean - asr_quantum_clean
    if abs(denominator) < 1e-9:
        return float("nan")
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# CQRS -- Composite Quantum Resilience Score
# ---------------------------------------------------------------------------

DEFAULT_CQRS_WEIGHTS = {"qsfr": 0.25, "eer": 0.25, "ctr": 0.20, "nlcs": 0.15, "qar": 0.15}


def cqrs(qsfr_val: float, eer_val: float, ctr_val: float, nlcs_val: float, qar_val: float,
         weights: dict | None = None) -> dict:
    """
    Weighted composite of the five outcome metrics. Weights default to
    the values named in the design source (0.25/0.25/0.20/0.15/0.15)
    -- ASSERTED, NOT DERIVED. Nothing in the source or this codebase
    justifies these specific values (not PCA-derived, not fit to any
    criterion, not cited to an external source). Report CQRS in the
    paper alongside either (a) a stated justification for this weighting,
    or (b) a sensitivity analysis showing the qualitative conclusion
    doesn't depend on the specific weights (e.g. equal weights, or a
    weight-perturbation sweep) -- a single composite number computed
    from unjustified weights invites exactly the "where did that number
    come from" question a defined-but-unexplained formula gets asked.

    If ctr_val is NaN (the common case: CTR is not computable at the
    primary 20-qubit configuration -- see this module's docstring),
    the composite is renormalized over the remaining four metrics
    rather than silently propagating NaN through the whole score or
    silently substituting a placeholder value for the missing term.
    """
    w = dict(weights) if weights else dict(DEFAULT_CQRS_WEIGHTS)
    values = {"qsfr": qsfr_val, "eer": eer_val, "ctr": ctr_val, "nlcs": nlcs_val, "qar": qar_val}

    available = {k: v for k, v in values.items() if np.isfinite(v)}
    missing = [k for k, v in values.items() if not np.isfinite(v)]

    if not available:
        return {"CQRS": float("nan"), "weights_used": {}, "missing_metrics": missing}

    w_available = {k: w[k] for k in available}
    w_total = sum(w_available.values())
    w_normalized = {k: v / w_total for k, v in w_available.items()} if w_total > 0 else w_available

    score = sum(w_normalized[k] * available[k] for k in available)
    return {
        "CQRS": float(score),
        "weights_used": w_normalized,
        "missing_metrics": missing,
        "renormalized": len(missing) > 0,
    }


# ---------------------------------------------------------------------------
# NLCS -- Non-Local Correlation Stability (rescoped to a reduced subsystem;
# see module docstring for why the full-system + sampling design doesn't work)
# ---------------------------------------------------------------------------

def _non_local_correlation_strength(rho: np.ndarray) -> float:
    """C = sum of |rho_ij| over index pairs (i, j) whose bitstring
    representations differ at >=2 bit positions (Hamming distance >= 2),
    i.e. genuinely non-local coherences, excluding single-bit-flip terms.
    Full O(d^2) double loop -- tractable because `rho` here is a REDUCED
    density matrix (subsystem size matching EER's partial-trace cut,
    typically ~10 qubits -> d=1024, d^2 ~= 10^6), not the full system."""
    d = rho.shape[0]
    total = 0.0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if bin(i ^ j).count("1") >= 2:
                total += abs(rho[i, j])
    return float(total)


def nlcs(rho_clean_reduced: np.ndarray, rho_attacked_reduced: np.ndarray) -> float:
    """
    NLCS = C(rho_attacked_reduced) / C(rho_clean_reduced).

    Both inputs must already be REDUCED density matrices (e.g. from
    src.quantum.tomography.reduced_density_matrix), not full-system
    states -- see module docstring. This is a genuine rescoping of the
    metric to "non-local correlation stability within the traced-out
    subsystem," not an approximation of a full-system quantity that
    can't actually be computed at this study's qubit count.
    """
    if rho_clean_reduced.shape != rho_attacked_reduced.shape:
        raise ValueError(
            f"Clean and attacked reduced density matrices must be the same shape "
            f"(same subsystem cut); got {rho_clean_reduced.shape} vs {rho_attacked_reduced.shape}."
        )
    c_clean = _non_local_correlation_strength(rho_clean_reduced)
    c_attacked = _non_local_correlation_strength(rho_attacked_reduced)
    if c_clean < 1e-12:
        return float("nan")
    return float(c_attacked / c_clean)


# ---------------------------------------------------------------------------
# CTR -- Coherence Time Retention (hard-gated: only runs where a dense
# density matrix is actually tractable; see module docstring)
# ---------------------------------------------------------------------------

MAX_QUBITS_FOR_DENSITY_MATRIX = 12  # 4^12 ~= 16.8M entries, ~270MB at complex128 -- tractable
BYTES_PER_COMPLEX128 = 16


def _density_matrix_memory_bytes(n_qubits: int) -> int:
    dim = 2 ** n_qubits
    return dim * dim * BYTES_PER_COMPLEX128


def check_ctr_tractable(n_qubits: int) -> None:
    """
    Raises a clear, specific error if n_qubits exceeds what a dense
    density-matrix representation can realistically hold in memory --
    called at the START of ctr() so a 20-qubit call fails immediately
    with an explanatory message, rather than attempting to allocate a
    ~17.6 TB array and hanging or crashing with an opaque MemoryError.
    """
    if n_qubits > MAX_QUBITS_FOR_DENSITY_MATRIX:
        required_gb = _density_matrix_memory_bytes(n_qubits) / (1024 ** 3)
        max_gb = _density_matrix_memory_bytes(MAX_QUBITS_FOR_DENSITY_MATRIX) / (1024 ** 3)
        raise ValueError(
            f"CTR is not computable at n_qubits={n_qubits}: a dense density matrix "
            f"needs 4^{n_qubits} = {2**n_qubits}x{2**n_qubits} complex128 entries "
            f"(~{required_gb:,.1f} GB). This study's primary configuration (20 qubits) "
            f"requires ~{_density_matrix_memory_bytes(20) / (1024**4):,.1f} TB, which is "
            f"not a performance issue to optimize -- it is a hard memory ceiling. CTR "
            f"is only computable here at n_qubits <= {MAX_QUBITS_FOR_DENSITY_MATRIX} "
            f"(~{max_gb:.2f} GB at the ceiling). Report CTR, if at all, on a reduced-qubit "
            f"ablation explicitly scoped and labeled as such -- not as a primary-configuration result."
        )


def _coherence_measure(rho: np.ndarray) -> float:
    """Proxy for T2 in a noiseless-simulator setting: sum of |off-diagonal|
    magnitude. This is a PROXY, not a physical coherence TIME -- it requires
    a genuine sequence of states under some decoherence process to fit a
    decay constant against (see ctr()'s docstring for what produces that
    sequence here); a single density matrix alone gives one coherence
    VALUE, not a time."""
    diag = np.diag(np.diag(rho))
    off_diagonal = rho - diag
    return float(np.sum(np.abs(off_diagonal)))


def fit_decay_constant(coherence_series: np.ndarray, dt: float = 1.0) -> float:
    """Fits coherence(t) = A * exp(-t / T2) via nonlinear least squares
    and returns T2. Requires scipy (already a project dependency)."""
    from scipy.optimize import curve_fit

    t = np.arange(len(coherence_series)) * dt

    def model(t, amplitude, decay_constant):
        return amplitude * np.exp(-t / np.maximum(decay_constant, 1e-12))

    coherence_series = np.asarray(coherence_series, dtype=float)
    window = dt * len(coherence_series)
    # Identifiability guard: if the series shows no measurable decay (e.g. a
    # noise-free run, or too few steps to see any), T2 is unidentifiable and
    # the fit returns an arbitrarily large number. A ratio of two such fits
    # produced CTR values of 1e7 to 1e8 that then dominated CQRS. Return NaN
    # instead; cqrs() already renormalises its weights when CTR is missing.
    scale = float(np.max(np.abs(coherence_series))) if len(coherence_series) else 0.0
    if scale <= 0.0 or float(np.ptp(coherence_series)) <= 1e-6 * scale:
        return float("nan")
    initial_guess = [coherence_series[0], window / 2]
    try:
        popt, _ = curve_fit(model, t, coherence_series, p0=initial_guess, maxfev=10000)
        t2 = float(popt[1])
    except Exception:
        return float("nan")
    # A T2 more than 50x the observed window means the data carry no
    # information about the decay time.
    if not np.isfinite(t2) or t2 <= 0.0 or t2 > 50.0 * window:
        return float("nan")
    return t2


def ctr(rho_clean_series: list[np.ndarray], rho_attacked_series: list[np.ndarray],
        n_qubits: int, dt: float = 1.0) -> float:
    """
    CTR = T2_attacked / T2_clean, fit from a coherence-decay series.

    n_qubits is required explicitly (not inferred from the matrices'
    shape alone) so check_ctr_tractable() can run BEFORE any large
    allocation is attempted, even if the caller already has the
    (small, already-tractable) matrices in hand -- this keeps the
    tractability check from being silently skippable by construction.

    rho_*_series: a list of density matrices at successive simulated
    time steps under SOME decoherence process. Generating that series
    requires a decoherence-channel simulator (e.g. PennyLane's
    default.mixed device with AmplitudeDamping/PhaseDamping channels)
    that this codebase does not currently build anywhere -- this
    function computes CTR correctly GIVEN such a series, but does not
    itself construct one. That is a separate, not-yet-built piece of
    work, distinct from (and in addition to) the tractability ceiling
    above.
    """
    check_ctr_tractable(n_qubits)

    coherence_clean = np.array([_coherence_measure(r) for r in rho_clean_series])
    coherence_attacked = np.array([_coherence_measure(r) for r in rho_attacked_series])

    t2_clean = fit_decay_constant(coherence_clean, dt)
    t2_attacked = fit_decay_constant(coherence_attacked, dt)

    if not np.isfinite(t2_clean) or t2_clean < 1e-12:
        return float("nan")
    return float(t2_attacked / t2_clean)
