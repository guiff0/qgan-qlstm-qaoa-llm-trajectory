"""PQCRunner's VJP must equal (full adjoint Jacobian)^T @ g -- the quantity the
old multi-observable QNode produced. Torch-free."""
import numpy as np
import pennylane as qml
import pytest

from src.quantum.pqc_runner import PQCRunner, _apply_ansatz


@pytest.mark.parametrize("ent", ["ring", "linear", "full"])
def test_vjp_matches_full_jacobian(ent):
    nq, nl = 5, 2
    rng = np.random.default_rng(0)
    theta = rng.normal(size=nq * nl * 3) * 0.3
    inputs = rng.normal(size=(3, nq))
    g = rng.normal(size=(3, nq))

    runner = PQCRunner(nq, nl, ent, "default.qubit")
    dev = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev, diff_method="backprop")
    def ref(x, w):
        _apply_ansatz(x, w, nq, nl, ent)
        return [qml.expval(qml.PauliZ(i)) for i in range(nq)]

    # forward agrees with the multi-observable reference
    fwd = runner.forward(inputs, theta)
    for b in range(3):
        assert np.allclose(fwd[b], np.asarray(ref(inputs[b], theta)), atol=1e-8)

    # VJP agrees with J^T g computed from the reference's full Jacobian
    gx, gt = runner.vjp(inputs, theta, g, need_input_grad=True)
    exp_t = np.zeros_like(theta)
    for b in range(3):
        jt = qml.jacobian(lambda w: qml.math.stack(ref(inputs[b], w)))(qml.numpy.array(theta, requires_grad=True))
        exp_t += np.asarray(jt).T @ g[b]
        jx = qml.jacobian(lambda x: qml.math.stack(ref(x, theta)))(qml.numpy.array(inputs[b], requires_grad=True))
        assert np.allclose(gx[b], np.asarray(jx).T @ g[b], atol=1e-7)
    assert np.allclose(gt, exp_t, atol=1e-7)
