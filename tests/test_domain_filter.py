import numpy as np
import pandas as pd

from src.data.preprocessing import clean_prices, domain_implausible_mask


def test_zero_price_is_flagged():
    df = pd.DataFrame({"close": [1.10, 1.101, 0.0, 1.102, 1.103]})
    mask = domain_implausible_mask(df, ["close"])
    assert mask.iloc[2]
    assert not mask.iloc[0]


def test_large_return_is_flagged():
    df = pd.DataFrame({"close": [1.10, 1.101, 1.102, 2.5, 1.105, 1.106]})
    mask = domain_implausible_mask(df, ["close"])
    assert mask.iloc[3]  # the spike itself
    assert not mask.iloc[1]


def test_normal_returns_not_flagged():
    df = pd.DataFrame({"close": [1.10, 1.1005, 1.101, 1.0995, 1.102]})
    mask = domain_implausible_mask(df, ["close"])
    assert not mask.any()


def test_clean_prices_interpolates_through_domain_violations():
    df = pd.DataFrame({"close": [1.10, 1.101, 0.0, 1.103, 1.104]})
    cleaned = clean_prices(df, ["close"])
    assert cleaned["close"].iloc[2] > 1.0  # interpolated, not left at 0
    assert np.isclose(cleaned["close"].iloc[2], 1.102, atol=0.01)


def test_custom_threshold_is_respected():
    df = pd.DataFrame({"close": [1.10, 1.101, 1.102, 1.15, 1.103]})
    loose = domain_implausible_mask(df, ["close"], max_abs_return=0.5)
    tight = domain_implausible_mask(df, ["close"], max_abs_return=0.02)
    assert not loose.any()
    assert tight.any()
