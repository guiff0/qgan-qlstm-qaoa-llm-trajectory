"""The new `rzz:<theta>` topology under REAL PennyLane (the other entangler tests use a stand-in).

    python -m pytest tests/test_rzz_topology.py -q -s        # -s prints the entropy table at the end

What it proves
  * rzz:0.0 is exactly the zero-entanglement circuit ("none"), in the training runner and in the state used for tomography
  * the circuit that TRAINS (pqc_runner._apply_ansatz) and the circuit that is MEASURED (circuits.apply_circuit_gates_only)
    produce the same state for ring and for rzz, so an entropy dose calibrated on one describes the other
  * IsingZZ works with the fast runner's adjoint/VJP path on default.qubit and, if installed, lightning.qubit
  * the last-layer RZZ is skipped (a Z read-out cannot see it) and earlier layers' RZZ does change the read-outs
  * bipartite entropy is ~0 at theta = 0 and positive for theta > 0 (the ordering across theta is what the Stage 1 gate measures)
"""
import math

import numpy as np
import pennylane as qml
import pytest

from src.quantum.circuits import apply_circuit_gates_only
from src.quantum.pqc_runner import PQCRunner, _apply_ansatz
from src.quantum.tomography import reduced_density_matrix, statevector_from_circuit, von_neumann_entropy

NQ = 5


def _devices():
    out = ["default.qubit"]
    try:
        qml.device("lightning.qubit", wires=2)
        out.append("lightning.qubit")
    except Exception:  # lightning not installed: the default.qubit case still runs
        pass
    return out


DEVICES = _devices()


def _state_train(inputs, w, nq, nl, ent):
    dev = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev)
    def f():
        _apply_ansatz(inputs, w, nq, nl, ent)
        return qml.state()
    return np.asarray(f())


def _state_measured(inputs, w, nq, nl, ent):
    return statevector_from_circuit(apply_circuit_gates_only(inputs, w, nq, nl, ent), w, nq)


def _draw(nq, nl, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=nq), r.normal(size=nq * nl * 3) * 0.5


def test_rzz_zero_is_exactly_the_zero_entanglement_circuit():
    for nl in (1, 3):
        x, w = _draw(NQ, nl)
        assert np.allclose(_state_train(x, w, NQ, nl, "rzz:0.0"), _state_train(x, w, NQ, nl, "none"), atol=1e-12)
        assert np.allclose(_state_measured(x, w, NQ, nl, "rzz:0.0"), _state_measured(x, w, NQ, nl, "none"), atol=1e-12)
    rho = reduced_density_matrix(_state_train(*_draw(NQ, 3), NQ, 3, "rzz:0.0"), NQ, [0, 1])
    assert von_neumann_entropy(rho) < 1e-6


@pytest.mark.parametrize("ent", ["ring", "full", "linear", "none", "rzz:0.6"])
def test_training_circuit_and_measured_circuit_are_the_same_state(ent):
    x, w = _draw(NQ, 3, seed=1)
    assert np.allclose(_state_train(x, w, NQ, 3, ent), _state_measured(x, w, NQ, 3, ent), atol=1e-10)


@pytest.mark.parametrize("dev_name", DEVICES)
def test_runner_forward_and_vjp_match_reference_for_rzz(dev_name):
    nl = 3
    rng = np.random.default_rng(0)
    theta = rng.normal(size=NQ * nl * 3) * 0.3
    inputs = rng.normal(size=(3, NQ))
    g = rng.normal(size=(3, NQ))
    ent = "rzz:0.6"
    runner = PQCRunner(NQ, nl, ent, dev_name)
    dev = qml.device("default.qubit", wires=NQ)

    @qml.qnode(dev, diff_method="backprop")
    def ref(x, w):
        _apply_ansatz(x, w, NQ, nl, ent)
        return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]

    fwd = runner.forward(inputs, theta)
    for b in range(3):
        assert np.allclose(fwd[b], np.asarray(ref(inputs[b], theta)), atol=1e-8)
    gx, gt = runner.vjp(inputs, theta, g, need_input_grad=True)
    exp_t = np.zeros_like(theta)
    for b in range(3):
        jt = qml.jacobian(lambda w: qml.math.stack(ref(inputs[b], w)))(qml.numpy.array(theta, requires_grad=True))
        exp_t += np.asarray(jt).T @ g[b]
        jx = qml.jacobian(lambda x: qml.math.stack(ref(x, theta)))(qml.numpy.array(inputs[b], requires_grad=True))
        assert np.allclose(gx[b], np.asarray(jx).T @ g[b], atol=1e-7)
    assert np.allclose(gt, exp_t, atol=1e-7)


def test_last_layer_rzz_is_skipped_and_earlier_layers_matter():
    x, w = _draw(NQ, 1)
    one = PQCRunner(NQ, 1, "rzz:0.9", "default.qubit").forward(x[None, :], w)
    none1 = PQCRunner(NQ, 1, "none", "default.qubit").forward(x[None, :], w)
    assert np.allclose(one, none1, atol=1e-12)                       # a single (= last) layer: no entangler applied
    x3, w3 = _draw(NQ, 3)
    a = PQCRunner(NQ, 3, "rzz:0.9", "default.qubit").forward(x3[None, :], w3)
    b = PQCRunner(NQ, 3, "none", "default.qubit").forward(x3[None, :], w3)
    assert not np.allclose(a, b, atol=1e-6)                          # earlier layers' RZZ reaches the Z read-outs


def test_entropy_dose_table_is_zero_at_theta_zero_and_positive_after():
    nq, nl = 6, 3
    thetas = [0.0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2]
    rows = []
    for th in thetas:
        s = []
        for seed in range(5):
            x, w = _draw(nq, nl, seed=seed)
            rho = reduced_density_matrix(_state_measured(x, w, nq, nl, f"rzz:{th!r}"), nq, list(range(nq // 2)))
            s.append(von_neumann_entropy(rho))
        rows.append((th, float(np.mean(s)), float(np.std(s))))
    print("\ntheta      mean S_E   sd over 5 random draws   (random weights, NOT trained generators)")
    for th, m, sd in rows:
        print(f"{th:8.4f}   {m:8.4f}   {sd:8.4f}")
    assert rows[0][1] < 1e-6 and all(m > 1e-3 for _, m, _ in rows[1:])
