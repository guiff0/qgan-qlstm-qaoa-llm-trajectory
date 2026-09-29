import math

from src.evaluation.talis import TaLISConfig, compute_talis


def test_basic_formula_default_weights():
    # asr=0.10 (fraction), latency=40ms == reference -> ratio 1.0
    score = compute_talis(0.10, 40.0)
    assert math.isclose(score, 0.5 * 0.10 + 0.5 * 1.0)


def test_asr_percentage_is_normalized():
    # 8.7 must be read as 8.7%, i.e. 0.087, same as passing 0.087 directly
    a = compute_talis(8.7, 40.0)
    b = compute_talis(0.087, 40.0)
    assert math.isclose(a, b)


def test_latency_cap_bounds_the_score():
    cfg = TaLISConfig(latency_reference_ms=40.0, latency_cap=5.0)
    huge_latency = compute_talis(0.0, 100_000.0, cfg)
    at_cap = compute_talis(0.0, 40.0 * 5.0, cfg)
    assert math.isclose(huge_latency, at_cap)
    assert math.isclose(huge_latency, 0.5 * 5.0)


def test_missing_or_nan_inputs_return_nan():
    assert math.isnan(compute_talis(None, 40.0))
    assert math.isnan(compute_talis(0.1, None))
    assert math.isnan(compute_talis(float("nan"), 40.0))
    assert math.isnan(compute_talis(0.1, float("nan")))


def test_custom_weights():
    cfg = TaLISConfig(impact_weight=0.8, latency_weight=0.2, latency_reference_ms=40.0, latency_cap=5.0)
    score = compute_talis(1.0, 40.0, cfg)  # asr=100%, latency at reference
    assert math.isclose(score, 0.8 * 1.0 + 0.2 * 1.0)


def test_lower_is_better_ordering():
    # a model with lower ASR and lower latency should score lower (less risk)
    safer = compute_talis(0.05, 20.0)
    riskier = compute_talis(0.30, 80.0)
    assert safer < riskier
