"""Cheap pre-check for the entanglement-dose design: how does half-register entropy depend on theta and on the weight scale?

    python -m scripts.simulate_entropy_dose --n-qubits 12 --layers 4 --weight-sd 0.1 0.3 1.0 --draws 40

Pure numpy (no PennyLane), same gate order as the training circuit: RY/RZ encoding, then per layer RX RY RZ on every qubit and,
for every layer EXCEPT the last, IsingZZ(theta) on every ring edge. Random weights, NOT trained generators: it says whether the
dose CAN be monotone at a given weight scale, not what your trained models do (Stage 1 measures that).
Reports the mean curve, the spread, how often a single draw is strictly increasing, and Spearman rho of the mean curve.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from scipy import stats


def _apply1(psi, U, q, n):
    psi = np.moveaxis(psi.reshape([2] * n), q, 0)
    return np.moveaxis(np.tensordot(U, psi, axes=([1], [0])), 0, q).reshape(-1)


def _rx(a): c, s = math.cos(a / 2), math.sin(a / 2); return np.array([[c, -1j * s], [-1j * s, c]])
def _ry(a): c, s = math.cos(a / 2), math.sin(a / 2); return np.array([[c, -s], [s, c]], dtype=complex)
def _rz(a): return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])


def _zz_ring(psi, theta, n):
    idx = np.arange(2 ** n)
    bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
    ph = np.zeros(2 ** n)
    for q in range(n):
        ph += np.where(bits[q] == bits[(q + 1) % n], -theta / 2, theta / 2)
    return psi * np.exp(1j * ph)


def state(n, layers, theta, x, w, skip_last=True):
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for i in range(n):
        psi = _apply1(_apply1(psi, _ry(x[i]), i, n), _rz(0.1 * x[i]), i, n)
    for l in range(layers):
        for q in range(n):
            k = l * n * 3 + q * 3
            for g, a in ((_rx, w[k]), (_ry, w[k + 1]), (_rz, w[k + 2])):
                psi = _apply1(psi, g(a), q, n)
        if theta > 0 and not (skip_last and l == layers - 1):
            psi = _zz_ring(psi, theta, n)
    return psi


def half_entropy(psi, n):
    s = np.linalg.svd(psi.reshape(2 ** (n // 2), -1), compute_uv=False) ** 2
    s = s[s > 1e-12]
    return float(-(s * np.log2(s)).sum())


def curve(n, layers, thetas, weight_sd, draws, seed0=0):
    R = []
    for d in range(draws):
        r = np.random.default_rng(seed0 + d)
        x = r.normal(size=n)
        w = r.normal(size=n * layers * 3) * weight_sd
        R.append([half_entropy(state(n, layers, t, x, w), n) for t in thetas])
    return np.array(R)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-qubits", type=int, default=12)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--weight-sd", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--thetas", type=float, nargs="+", default=[0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2])
    ap.add_argument("--draws", type=int, default=40)
    a = ap.parse_args()
    print("theta grid:", [round(t, 4) for t in a.thetas])
    for sd in a.weight_sd:
        R = curve(a.n_qubits, a.layers, a.thetas, sd, a.draws)
        m = R.mean(0)
        rho = stats.spearmanr(a.thetas, m).statistic
        print(f"weight sd {sd:<5} mean S_E {np.round(m, 3)}  sd {np.round(R.std(0), 3)}  "
              f"mean curve increasing: {bool((np.diff(m) > 0).all())}  rho(theta, mean S_E) = {rho:+.2f}  "
              f"single draw strictly increasing: {(np.diff(R, axis=1) > 0).all(axis=1).mean():.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
