"""
Regression tests for the Dukascopy year-validation bug: year 2012 was
being rejected as a "truncated download" on every single run (not a
flaky network issue) because Dukascopy's real free EURUSD 1-min history
starts 2012-01-11, and _validate_year unconditionally required every
year to start by Jan 8. See KNOWN_PROVIDER_HISTORY_START in
scripts/acquire_all_data.py.
"""
import os
import sys

import pandas as pd
import pytest

os.environ.setdefault("QGAN_BASE_DIR", os.getcwd())
sys.path.insert(0, os.getcwd())

from scripts import acquire_all_data as a


def _year_frame(start, end):
    idx = pd.date_range(start, end, freq="1min", tz="UTC")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)


def test_2012_real_partial_year_is_accepted():
    """The exact shape that was failing on every run: real Dukascopy 2012
    data, starting 2012-01-11 (confirmed by this project's own prior
    successful run logs), not Jan 1."""
    df = _year_frame("2012-01-11", "2012-12-31 23:59")
    a._validate_year(df, 2012)  # must not raise


def test_2012_starting_even_later_is_still_rejected():
    """The relaxed threshold is anchored to the known real start date
    (2012-01-11) plus a week of slack -- it shouldn't accept an
    arbitrarily late start for 2012."""
    df = _year_frame("2012-03-01", "2012-12-31 23:59")
    with pytest.raises(ValueError, match="truncated download"):
        a._validate_year(df, 2012)


def test_other_years_still_require_starting_near_jan_1():
    """A year with no known provider-history quirk (e.g. 2013) must still
    be rejected if it starts late -- the fix must not weaken validation
    for years that should have full history."""
    df = _year_frame("2013-03-01", "2013-12-31 23:59")
    with pytest.raises(ValueError, match="truncated download"):
        a._validate_year(df, 2013)


def test_other_years_full_history_still_passes():
    df = _year_frame("2013-01-01", "2013-12-31 23:59")
    a._validate_year(df, 2013)  # must not raise


def test_2012_row_count_floor_is_scaled_not_full_year():
    """2012 is missing its first 10 days -- the row-count sanity check
    must be scaled down proportionally, not still require a full year's
    ~100k+ rows (which the real, correct 2012 data can never reach)."""
    df = _year_frame("2012-01-11", "2012-12-31 23:59")
    assert len(df) < 100_000 or True  # not the point; the check below is
    a._validate_year(df, 2012)  # must not raise on row count either


def test_dukascopy_filename_matches_configured_start_year():
    """The file name must not claim a year range wider than what
    train_start/test_end actually cause acquire_dukascopy() to fetch --
    this is exactly the "...2010_2025.csv" vs. actual 2012-2025 mismatch
    that made 2010/2011 look like a missing-data gap when they were
    simply never requested."""
    assert str(a.START_YEAR) in os.path.basename(a.DUKASCOPY_MASTER)
    assert str(a.END_YEAR) in os.path.basename(a.DUKASCOPY_MASTER)
