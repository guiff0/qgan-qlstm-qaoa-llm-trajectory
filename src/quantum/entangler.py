"""One definition of the entangling block, as a gate list (pure python, testable without PennyLane).

NOT WIRED IN YET. The ansatz's entangling block is copy-pasted in circuits.py (x2), pqc_runner.py, gradient_obfuscation.py,
encoding_fidelity.py, attacks/quantum_attacks.py and feature_map.py. Adding a tunable-angle topology consistently means
replacing all of those with `apply_entangler`, then running tests/test_pqc_runner.py on a machine with PennyLane, because a
silent mismatch between the trained circuit and the circuit used for entropy would invalidate the dose calibration.

Topology strings: "ring" | "full" | "linear" | "none" (existing, CNOT) and "rzz:<theta>" (new): IsingZZ(theta) on every ring
edge INSTEAD of CNOTs. theta = 0 emits no two-qubit gate at all, so it is exactly "none" (the zero-dose anchor). theta is a
fixed design value, never trained.

READ-OUT CAVEAT: the generator reads PauliZ expectations and IsingZZ is diagonal in Z, so an rzz block in the LAST layer cannot
change those read-outs; it only raises the measured state entropy. Skip the last-layer entangler (skip_last=True) or the
entropy dose will decouple from anything the generator outputs.
"""
from __future__ import annotations


def parse_entanglement(spec: str) -> tuple[str, float]:
    if isinstance(spec, str) and spec.startswith("rzz:"):
        return "rzz", float(spec.split(":", 1)[1])
    if spec in ("ring", "full", "linear", "none"):
        return spec, 0.0
    raise ValueError(f"Unknown entanglement topology: {spec}")


def entangler_gates(spec: str, n_qubits: int) -> list[tuple]:
    """[(name, wires, param_or_None), ...] for ONE layer's entangling block."""
    kind, theta = parse_entanglement(spec)
    if kind == "none" or (kind == "rzz" and theta == 0.0):
        return []
    if kind == "ring":
        return [("CNOT", (q, (q + 1) % n_qubits), None) for q in range(n_qubits)]
    if kind == "full":
        return [("CNOT", (i, j), None) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if kind == "linear":
        return [("CNOT", (q, q + 1), None) for q in range(n_qubits - 1)]
    return [("IsingZZ", (q, (q + 1) % n_qubits), theta) for q in range(n_qubits)]


def apply_entangler(spec: str, n_qubits: int, layer: int, n_layers: int, skip_last: bool = True) -> None:
    import pennylane as qml  # imported here so the gate lists above test without PennyLane
    if skip_last and parse_entanglement(spec)[0] == "rzz" and layer == n_layers - 1:
        return
    for name, wires, p in entangler_gates(spec, n_qubits):
        qml.CNOT(wires=list(wires)) if name == "CNOT" else qml.IsingZZ(p, wires=list(wires))
