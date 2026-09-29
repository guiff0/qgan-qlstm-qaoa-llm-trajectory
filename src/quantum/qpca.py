"""
Quantum Principal Component Analysis (QPCA) via quantum phase estimation,
whose phase read-out is the inverse Quantum Fourier Transform.

======================================================================
WHAT THIS IS (and the honest scope of "quantum" here)
======================================================================
Lloyd, Mohseni & Rebentrost (2014) QPCA, in four steps:

  1. Encode the data covariance as a density matrix
         rho = C / tr(C)          (C = feature covariance, PSD, so rho >= 0, tr = 1)
  2. Density-matrix exponentiation: implement U = exp(2*pi*i*rho).
     Its eigenvalues are exp(2*pi*i*lambda_j) with lambda_j in [0,1]
     the eigenvalues of rho, sharing rho's eigenvectors v_j.
  3. Quantum phase estimation (QPE) of U on a copy of rho:
       clock register (n_clock qubits) in |+>^n
       controlled-U^(2^k) from clock qubit k onto the system register
       INVERSE QFT on the clock register           <-- the QFT step
     Measuring the clock returns the integer m ~ lambda_j * 2^n_clock
     with probability ~ lambda_j (the input is rho itself), and the
     system register collapses onto v_j.
  4. Read the top eigenpairs from the clock histogram and the
     clock-conditioned system state.

SIMULATION CAVEATS -- read before citing as "quantum advantage":
  * Step 2 is done here with an exact classical matrix exponential
    (np linalg), NOT sample-based density-matrix exponentiation on
    copies of rho. On hardware, step 2 is where the claimed exponential
    speedup lives and it needs rho to be low-rank and loadable (QRAM).
  * rho is prepared through its purification (StatePrep on system +
    reference registers), a classical state-vector shortcut.
  * Total qubits = n_clock + 2*n_sys. This is a CLASSICAL simulation
    that costs O(2^qubits); it gives no speedup, it verifies the
    algorithm and lets you compare against classical PCA.
  * Eigenvalue resolution is 2^-n_clock. Eigenvalues below that land in
    bin 0; increase n_clock or lower dimension if the spectrum tail
    matters. Components are returned only for well-resolved peaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pennylane as qml


@dataclass
class QPCAConfig:
    n_components: int = 2
    n_clock: int = 6            # phase-estimation qubits (resolution 2**-n_clock)
    quantum_device: str = "default.qubit"
    peak_threshold: float = 0.01  # min clock-bin probability to count as a spectral peak
    standardize: bool = True


@dataclass
class QPCAResult:
    eigenvalues: np.ndarray            # estimated eigenvalues of rho, descending
    eigenvectors: np.ndarray           # (n_features, k) columns, unit norm, sign-fixed
    explained_variance_ratio: np.ndarray
    clock_distribution: np.ndarray     # p(m) over 2**n_clock bins
    classical_eigenvalues: np.ndarray  # exact, for verification
    classical_eigenvectors: np.ndarray
    eigenvalue_max_abs_error: float
    eigenvector_abs_overlap: np.ndarray  # |<v_quantum, v_classical>| per component (1.0 = perfect)
    n_qubits_simulated: int
    mean_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale_: np.ndarray = field(default_factory=lambda: np.ones(0))


def density_matrix_from_data(X: np.ndarray, standardize: bool = True):
    """Covariance -> trace-normalized density matrix, zero-padded to 2**n_sys."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    scale = Xc.std(axis=0)
    scale[scale == 0] = 1.0
    if standardize:
        Xc = Xc / scale
    else:
        scale = np.ones_like(scale)
    C = (Xc.T @ Xc) / max(len(X) - 1, 1)
    d = C.shape[0]
    n_sys = max(1, int(np.ceil(np.log2(d))))
    D = 2 ** n_sys
    rho = np.zeros((D, D))
    rho[:d, :d] = C
    rho /= np.trace(rho)
    return rho, n_sys, d, mean, scale


class QuantumPCA:
    def __init__(self, config: QPCAConfig = None):
        self.config = config or QPCAConfig()

    # ---- circuit -------------------------------------------------------
    def _build_qnode(self, rho: np.ndarray, n_sys: int):
        cfg = self.config
        n_c = cfg.n_clock
        clock = list(range(n_c))
        system = list(range(n_c, n_c + n_sys))
        reference = list(range(n_c + n_sys, n_c + 2 * n_sys))

        # Purification of rho: |psi> = sum_j sqrt(l_j) |v_j>|v_j*>  ->  tracing out
        # `reference` leaves exactly rho on `system`. Equivalent closed form:
        # |psi> = vec(sqrt(rho)) (row-major), since rho is real symmetric.
        w, V = np.linalg.eigh(rho)
        w = np.clip(w, 0.0, None)
        sqrt_rho = (V * np.sqrt(w)) @ V.T
        psi = sqrt_rho.reshape(-1)
        psi = psi / np.linalg.norm(psi)

        # U^(2^k) = exp(2*pi*i*rho*2^k) via the shared eigendecomposition.
        powers = [(V * np.exp(2j * np.pi * w * (2 ** k))) @ V.conj().T for k in range(n_c)]

        dev = qml.device(cfg.quantum_device, wires=n_c + 2 * n_sys)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(psi, wires=system + reference)
            for q in clock:
                qml.Hadamard(wires=q)
            for k, q in enumerate(clock):
                # clock qubit 0 is the MOST significant bit for qml.QFT's convention,
                # so it controls the LARGEST power (2^(n_c-1)).
                power = powers[n_c - 1 - k]
                qml.ctrl(qml.QubitUnitary(power, wires=system), control=q)
            qml.adjoint(qml.QFT(wires=clock))      # <-- inverse Quantum Fourier Transform
            return qml.state()

        return circuit, n_c, n_sys

    # ---- fit -----------------------------------------------------------
    def fit(self, X: np.ndarray) -> QPCAResult:
        cfg = self.config
        rho, n_sys, d, mean, scale = density_matrix_from_data(X, cfg.standardize)
        circuit, n_c, n_sys = self._build_qnode(rho, n_sys)
        state = np.asarray(circuit()).reshape(2 ** n_c, 2 ** n_sys, 2 ** n_sys)
        # state[m, s, r]: clock bin m, system s, reference r
        p_clock = (np.abs(state) ** 2).sum(axis=(1, 2))

        # Peaks of the clock histogram (local maxima above threshold).
        n_bins = 2 ** n_c
        peaks = [m for m in range(n_bins)
                 if p_clock[m] >= cfg.peak_threshold
                 and p_clock[m] >= p_clock[(m - 1) % n_bins]
                 and p_clock[m] >= p_clock[(m + 1) % n_bins]]
        pairs = []
        for m in peaks:
            bins = [(m - 1) % n_bins, m, (m + 1) % n_bins]
            mass = p_clock[bins].sum()
            # circular-safe weighted phase around the peak
            offsets = np.array([-1, 0, 1])
            lam = ((m + (offsets * p_clock[bins]).sum() / mass) % n_bins) / n_bins
            sigma = np.zeros((2 ** n_sys, 2 ** n_sys), dtype=complex)
            for b in bins:
                block = state[b]                    # (system, reference)
                sigma += block @ block.conj().T     # trace out reference
            sigma /= max(np.trace(sigma).real, 1e-15)
            ev, evec = np.linalg.eigh(sigma)
            pairs.append((lam, evec[:, -1], mass))
        pairs.sort(key=lambda t: -t[0])
        pairs = pairs[: cfg.n_components]

        if not pairs:
            raise RuntimeError(
                "QPCA found no resolvable spectral peak. Lower peak_threshold or raise n_clock."
            )

        lam_q = np.array([p[0] for p in pairs])
        vec_q = np.stack([np.real(p[1])[:d] for p in pairs], axis=1)  # drop padding rows
        vec_q /= np.linalg.norm(vec_q, axis=0, keepdims=True)
        for j in range(vec_q.shape[1]):               # deterministic sign
            if vec_q[np.argmax(np.abs(vec_q[:, j])), j] < 0:
                vec_q[:, j] *= -1

        w_c, V_c = np.linalg.eigh(rho)
        order = np.argsort(-w_c)
        w_c, V_c = w_c[order], V_c[:d][:, order]
        k = vec_q.shape[1]
        overlaps = np.abs(np.sum(vec_q * V_c[:, :k] / np.linalg.norm(V_c[:, :k], axis=0), axis=0))

        return QPCAResult(
            eigenvalues=lam_q,
            eigenvectors=vec_q,
            explained_variance_ratio=lam_q,  # tr(rho)=1, so lambda_j IS the variance share
            clock_distribution=p_clock,
            classical_eigenvalues=w_c[:k],
            classical_eigenvectors=V_c[:, :k],
            eigenvalue_max_abs_error=float(np.max(np.abs(lam_q - w_c[:k]))),
            eigenvector_abs_overlap=overlaps,
            n_qubits_simulated=n_c + 2 * n_sys,
            mean_=mean,
            scale_=scale,
        )

    @staticmethod
    def transform(X: np.ndarray, result: QPCAResult) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - result.mean_) / result.scale_
        return Xs @ result.eigenvectors
