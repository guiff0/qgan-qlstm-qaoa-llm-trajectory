import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pennylane")

from src.baselines.qgan_llm import DATA_ENTROPY_N_INPUTS, data_reference_inputs


def test_data_inputs_are_fixed_and_independent_of_global_rng_state():
    np.random.seed(1)
    a = data_reference_inputs(12)
    np.random.seed(999)
    b = data_reference_inputs(12)
    assert a.shape == (DATA_ENTROPY_N_INPUTS, 12)
    assert np.array_equal(a, b)


def test_data_inputs_are_data_like_not_zero():
    x = data_reference_inputs(20)
    assert np.abs(x).max() > 0.5 and abs(x.std() - 1.0) < 0.3
