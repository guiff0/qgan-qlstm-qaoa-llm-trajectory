"""Pairwise mutual-information map (known states) and the figure script's skip/draw behaviour."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantum.entanglement_map import pairwise_mutual_information, save_entanglement_map  # noqa: E402


def _ghz(n):
    s = np.zeros(2 ** n, complex); s[0] = s[-1] = 1 / np.sqrt(2); return s


def test_product_state_has_zero_mutual_information():
    s = np.zeros(2 ** 4, complex); s[0] = 1
    r = pairwise_mutual_information(s, 4)
    assert np.allclose(r["mi"], 0) and np.allclose(r["single_entropy"], 0)


def test_bell_pair_is_two_bits_and_third_qubit_is_independent():
    bell = np.zeros(4, complex); bell[0] = bell[3] = 1 / np.sqrt(2)
    s = np.kron(bell, np.array([1, 0], complex))            # qubits 0,1 entangled; qubit 2 free
    r = pairwise_mutual_information(s, 3)
    assert np.isclose(r["mi"][0, 1], 2.0)
    assert np.isclose(r["mi"][0, 2], 0) and np.isclose(r["mi"][1, 2], 0)
    assert np.allclose(r["single_entropy"], [1, 1, 0])


def test_ghz_every_pair_is_one_bit_and_map_is_symmetric():
    r = pairwise_mutual_information(_ghz(5), 5)
    off = r["mi"][~np.eye(5, dtype=bool)]
    assert np.allclose(off, 1.0) and np.allclose(r["mi"], r["mi"].T) and np.allclose(np.diag(r["mi"]), 0)
    assert np.allclose(r["single_entropy"], 1.0)


def test_random_pure_state_mi_is_nonnegative_and_bounded():
    rng = np.random.default_rng(0)
    s = rng.normal(size=2 ** 6) + 1j * rng.normal(size=2 ** 6); s /= np.linalg.norm(s)
    mi = pairwise_mutual_information(s, 6)["mi"]
    assert mi.min() >= 0 and mi.max() <= 2.0 + 1e-9


def test_rejects_bad_statevectors():
    for bad in (np.ones(8, complex), np.array([1, 0, 0], complex)):
        try:
            pairwise_mutual_information(bad, 3)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_save_and_figure_script_roundtrip():
    from scripts import make_figures
    r = pairwise_mutual_information(_ghz(4), 4)
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "maps", "qgan_llm_seed42.json")
        save_entanglement_map(mp, r["mi"], r["single_entropy"],
                              {"model": "QGAN-LLM", "seed": 42, "entanglement_topology": "ring", "noise_strength": 0.01})
        assert json.load(open(mp))["mean_pairwise_mi"] == 1.0
        out = os.path.join(d, "figs")
        os.makedirs(out)
        assert make_figures.figure_31(os.path.join(d, "missing.json"), out) is None       # skipped, no crash
        q = os.path.join(d, "qpca.json")
        json.dump({"n_features": 3, "n_rows": 10, "qubits_simulated": 8, "quantum_eigenvalues": [2.0, 1.0],
                   "classical_eigenvalues": [2.01, 0.99], "eigenvalue_max_abs_error": 0.01,
                   "explained_variance_ratio": [0.6, 0.3],
                   "eigenvectors": [[1, 0, 0], [0, 1, 0]], "note": "n"}, open(q, "w"))     # stored (k, n_features)
        assert os.path.getsize(make_figures.figure_31(q, out)) > 1000
        figs = make_figures.figure_32(os.path.dirname(mp), out)
        assert len(figs) == 1 and os.path.getsize(figs[0]) > 1000
