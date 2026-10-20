"""Numpy-only tests: diagnostics, simulator parity with a dense reference, forecaster arms."""
import os, sys
from functools import reduce
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.advantage_diagnostics import (
    diebold_mariano, effective_rank, find_degenerate_groups, generator_rank_bound,
    lstm_generator_params, numerical_rank, qgan_generator_params, skill_vs_persistence,
    training_budget_fraction)
from src.quantum.feature_map import circuit_expvals
from src.baselines.feature_map_forecaster import FeatureMapForecaster


def test_degenerate_detection():
    assert find_degenerate_groups({"a": [0.5, 0.5], "b": [0.5]})["degenerate"]
    assert not find_degenerate_groups({"a": [0.5, 0.6], "b": [0.7]})["degenerate"]


def test_rank_bound_matches_linear_readout():
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 1, (500, 12))
    Y = Z @ rng.normal(size=(12, 32)) + rng.normal(size=32)
    assert numerical_rank(Y) == 12 == generator_rank_bound(12, 32)
    assert effective_rank(Y) <= 12.0 + 1e-9


def test_budget_and_params():
    b = training_budget_fraction(20, 16, 30, 3_715_921)
    assert b["max_distinct_rows"] == 9600 and abs(b["fraction_of_train_set"] - 0.00258) < 1e-4
    assert lstm_generator_params() == 87072 and qgan_generator_params() == 912


def test_skill_and_dm():
    rng = np.random.default_rng(1)
    cur = np.cumsum(rng.normal(size=2000)); nxt = cur + rng.normal(size=2000)
    assert abs(skill_vs_persistence(nxt, cur, cur)["skill"]) < 1e-12
    e = rng.normal(size=2000)
    assert diebold_mariano(e, e)["p_value"] == 1.0
    assert diebold_mariano(3 * e, e)["p_value"] < 1e-6


def _dense(x, w, n, L, topo):
    I, X, Zm = np.eye(2), np.array([[0, 1], [1, 0]]), np.diag([1, -1])
    RX = lambda t: np.array([[np.cos(t/2), -1j*np.sin(t/2)], [-1j*np.sin(t/2), np.cos(t/2)]])
    RY = lambda t: np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])
    RZ = lambda t: np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)])
    op = lambda U, q: reduce(np.kron, [U if i == q else I for i in range(n)])
    def cnot(c, t):
        P0, P1 = np.diag([1, 0]), np.diag([0, 1])
        return (reduce(np.kron, [P0 if i == c else I for i in range(n)])
                + reduce(np.kron, [P1 if i == c else (X if i == t else I) for i in range(n)]))
    psi = np.zeros(2**n, complex); psi[0] = 1
    for i in range(n):
        psi = op(RZ(.1*x[i]), i) @ (op(RY(x[i]), i) @ psi)
    for l in range(L):
        for q in range(n):
            b = l*n*3 + 3*q
            for U in (RX(w[b]), RY(w[b+1]), RZ(w[b+2])):
                psi = op(U, q) @ psi
        pairs = {"ring": [(q, (q+1) % n) for q in range(n)], "linear": [(q, q+1) for q in range(n-1)],
                 "none": []}[topo]
        for c, t in pairs:
            psi = cnot(c, t) @ psi
    return np.array([np.real(psi.conj() @ op(Zm, q) @ psi) for q in range(n)])


def test_simulator_matches_dense_reference():
    rng = np.random.default_rng(2)
    for n in (2, 3, 4):
        for topo in ("ring", "linear", "none"):
            w = rng.uniform(0, 6.28, 2*n*3); x = rng.normal(size=(3, n))
            ref = np.array([_dense(xi, w, n, 2, topo) for xi in x])
            assert np.abs(circuit_expvals(x, w, n, 2, topo) - ref).max() < 1e-10


def _series(n, signal, rng):
    X = rng.normal(size=(n + 1, 12)); r = 0.01 * rng.normal(size=n + 1)
    if signal:
        r = r + 0.01 * np.sin(2 * X[:, 0]) * np.cos(X[:, 1])
    return X, np.cumsum(np.r_[0, r[:-1]])


def test_no_signal_gives_no_skill_and_arms_run():
    rng = np.random.default_rng(3)
    tr, va, te = _series(3000, False, rng), _series(1000, False, rng), _series(2000, False, rng)
    for arm in ("linear", "classical_rf", "quantum"):
        r = FeatureMapForecaster(arm, n_qubits=4, n_layers=2, seed=0).fit(*tr, *va).evaluate(*te)
        assert abs(r["skill"]) < 0.01          # nothing to find beyond persistence
