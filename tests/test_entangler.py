import math, os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum.entangler import entangler_gates, parse_entanglement


def test_existing_topologies_match_the_cnot_counts_in_the_ansatz():
    n = 12
    assert len(entangler_gates("ring", n)) == n and len(entangler_gates("linear", n)) == n - 1
    assert len(entangler_gates("full", n)) == n * (n - 1) // 2 and entangler_gates("none", n) == []
    assert entangler_gates("ring", 4)[-1][1] == (3, 0)


def test_rzz_zero_is_exactly_the_zero_dose_anchor():
    assert entangler_gates("rzz:0.0", 12) == entangler_gates("none", 12) == []
    g = entangler_gates(f"rzz:{math.pi / 4}", 12)
    assert len(g) == 12 and all(x[0] == "IsingZZ" and x[2] == pytest.approx(math.pi / 4) for x in g)
    assert not any(x[0] == "CNOT" for x in g)


def test_unknown_spec_rejected():
    with pytest.raises(ValueError):
        parse_entanglement("star")


# ---- apply_entangler against a recording stand-in for PennyLane (no PennyLane needed) ----
import re
import types


def _recorder(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(CNOT=lambda wires: calls.append(("CNOT", tuple(wires))),
                                 IsingZZ=lambda th, wires: calls.append(("IsingZZ", tuple(wires), th)))
    monkeypatch.setitem(sys.modules, "pennylane", fake)
    return calls


def test_apply_entangler_reproduces_the_old_cnot_order(monkeypatch):
    from src.quantum.entangler import apply_entangler
    calls = _recorder(monkeypatch)
    apply_entangler("ring", 4, 0, 2)
    assert calls == [("CNOT", (0, 1)), ("CNOT", (1, 2)), ("CNOT", (2, 3)), ("CNOT", (3, 0))]
    calls.clear(); apply_entangler("full", 3, 1, 2)
    assert calls == [("CNOT", (0, 1)), ("CNOT", (0, 2)), ("CNOT", (1, 2))]
    calls.clear(); apply_entangler("linear", 3, 1, 2)
    assert calls == [("CNOT", (0, 1)), ("CNOT", (1, 2))]
    calls.clear(); apply_entangler("none", 3, 0, 2)
    assert calls == []


def test_rzz_is_skipped_in_the_last_layer_only_and_cnot_topologies_are_never_skipped(monkeypatch):
    from src.quantum.entangler import apply_entangler
    calls = _recorder(monkeypatch)
    apply_entangler("rzz:0.5", 4, 1, 2)                 # last layer
    assert calls == []
    apply_entangler("rzz:0.5", 4, 0, 2)
    assert len(calls) == 4 and all(c[0] == "IsingZZ" and c[2] == 0.5 for c in calls)
    calls.clear(); apply_entangler("ring", 4, 1, 2)     # last layer, CNOT topology: unchanged behaviour
    assert len(calls) == 4
    calls.clear(); apply_entangler("rzz:0.0", 4, 0, 2)
    assert calls == []


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = ["src/quantum/circuits.py", "src/quantum/pqc_runner.py", "src/quantum/gradient_obfuscation.py",
         "src/quantum/encoding_fidelity.py", "src/attacks/quantum_attacks.py"]


def test_no_site_keeps_its_own_copy_of_the_entangling_block_and_all_use_the_shared_one():
    for rel in SITES:
        src = open(os.path.join(ROOT, rel), encoding="utf8").read()
        assert not re.search(r'entanglement\s*==\s*"(ring|full|linear)"', src), f"{rel} still has its own topology chain"
        assert "apply_entangler" in src, f"{rel} does not use the shared entangler"
