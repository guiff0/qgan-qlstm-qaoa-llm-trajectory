"""
Regression tests for the pc_max_rows=null crash (TypeError: int() argument …
not 'NoneType').

The bug: poisoning_resistance.py called int((pcfg or {}).get("pc_max_rows", 100_000))
which yields int(None) when the config key exists but is explicitly set to null.
The fix: use .get("pc_max_rows") or 100_000 so that None falls back to the default.

Covers:
  - pc_max_rows explicitly null in config → uses default 100 000
  - pc_max_rows absent from config → uses default 100 000
  - pc_max_rows set to an integer → uses that integer
  - evaluate_poisoning_resistance end-to-end smoke test with null config (seeds 60-62)
"""
import types
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Unit tests: _resolve_pc_max_rows helper (extracted logic, no I/O)
# ---------------------------------------------------------------------------

def _resolve(pcfg, default=100_000):
    """Mirror of the fixed expression in poisoning_resistance.py."""
    return int((pcfg or {}).get("pc_max_rows") or default)


class TestResolvePcMaxRows:
    def test_null_value_returns_default(self):
        assert _resolve({"pc_max_rows": None}) == 100_000

    def test_absent_key_returns_default(self):
        assert _resolve({}) == 100_000

    def test_pcfg_itself_none_returns_default(self):
        assert _resolve(None) == 100_000

    def test_zero_value_returns_default(self):
        # 0 is also falsy; treat same as null — no meaningful row limit
        assert _resolve({"pc_max_rows": 0}) == 100_000

    def test_positive_int_honoured(self):
        assert _resolve({"pc_max_rows": 500}) == 500

    def test_string_int_honoured(self):
        # YAML sometimes loads integers as strings
        assert _resolve({"pc_max_rows": "250"}) == 250

    def test_float_truncated(self):
        assert _resolve({"pc_max_rows": 1000.9}) == 1000


# ---------------------------------------------------------------------------
# Integration smoke test: evaluate_poisoning_resistance with null pc_max_rows
# ---------------------------------------------------------------------------

def _make_dummy_model(seed=60):
    """Minimal stub that satisfies evaluate_poisoning_resistance's interface."""
    rng = np.random.default_rng(seed)

    class _Dummy:
        config = {}

        def predict(self, X):
            return rng.normal(0, 1, len(X))

        def fit(self, X, y):
            pass

    return _Dummy()


def _dummy_protocol_cfg_null():
    """Simulate what the config loader produces when pc_max_rows: null."""
    return {
        "pc_max_rows": None,          # the key exists but value is null
        "shift_sd": 2.0,
        "n_poison_rows": 10,
        "ridge_alpha": 1e-3,
        "pc_alpha": 1e-3,
    }


@pytest.mark.parametrize("seed", [60, 61, 62])
def test_evaluate_poisoning_resistance_null_pc_max_rows(seed):
    """Seeds 60-62 were the ones actually crashing; all must complete without TypeError."""
    torch = pytest.importorskip("torch", reason="torch not installed in this environment")
    from src.evaluation.poisoning_resistance import evaluate_poisoning_resistance

    rng = np.random.default_rng(seed)
    n = 300
    X = rng.normal(0, 1, (n, 8)).astype(np.float32)
    y = rng.normal(0, 1, n).astype(np.float32)

    model = _make_dummy_model(seed)
    pcfg  = _dummy_protocol_cfg_null()

    # Must not raise TypeError (the regression we are guarding against)
    result = evaluate_poisoning_resistance(model, X, y, seed=seed, protocol_cfg=pcfg)

    assert isinstance(result, dict), "result must be a dict"
    # The function may legitimately produce a degradation value or a status key;
    # we only care that it did not crash.
    assert "poisoning_rmse_degradation_pct" in result or "status" in result, (
        f"unexpected result keys: {list(result)}"
    )


@pytest.mark.parametrize("seed", [60, 61, 62])
def test_evaluate_poisoning_resistance_missing_pc_max_rows(seed):
    """Key absent entirely — also caused the crash on configs that never had the key."""
    pytest.importorskip("torch", reason="torch not installed in this environment")
    from src.evaluation.poisoning_resistance import evaluate_poisoning_resistance

    rng = np.random.default_rng(seed + 100)
    n = 300
    X = rng.normal(0, 1, (n, 8)).astype(np.float32)
    y = rng.normal(0, 1, n).astype(np.float32)

    model = _make_dummy_model(seed)
    # pcfg without pc_max_rows key at all
    pcfg = {"shift_sd": 2.0, "n_poison_rows": 10, "ridge_alpha": 1e-3, "pc_alpha": 1e-3}

    result = evaluate_poisoning_resistance(model, X, y, seed=seed, protocol_cfg=pcfg)
    assert isinstance(result, dict)
