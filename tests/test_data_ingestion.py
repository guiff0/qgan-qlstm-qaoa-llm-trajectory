"""
Tests for the API-ingestion path: src/data/data_loader.py and
scripts/acquire_all_data.py. No network access is needed -- yfinance and
`requests` are replaced with fakes.

Run:  python -m pytest tests/test_data_ingestion.py -q
"""
from __future__ import annotations

import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import data_loader as dl  # noqa: E402


def _bars(start, periods, freq="h"):
    ts = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1, "volume": 1.0})


def _vix(dates, vals):
    return pd.DataFrame({"date": pd.to_datetime(dates, utc=True), "vix": vals})


ZERO = dict(vix_lag_days=0, default_lag_days=0, release_lags={})


# ---------------------------------------------------------------- merge
def test_vix_close_not_visible_until_next_day():
    bars = _bars("2020-03-10 00:00", 72)                       # Mar 10, 11, 12
    vix = _vix(["2020-03-09", "2020-03-10", "2020-03-11"], [10.0, 20.0, 30.0])
    fred = pd.DataFrame({"date": pd.to_datetime(["2020-03-01"], utc=True), "X": [1.0]})
    out = dl.merge_all_sources(bars, fred, vix)
    by_ts = out.set_index("timestamp")["vix"]
    assert by_ts["2020-03-10 00:00"] == 10.0     # NOT day-of 20.0 (its close isn't known yet)
    assert by_ts["2020-03-10 23:00"] == 10.0
    assert by_ts["2020-03-11 00:00"] == 20.0
    assert by_ts["2020-03-12 00:00"] == 30.0


def test_monthly_cpi_applied_only_after_publication_lag():
    bars = _bars("2020-02-08", 24 * 12)                        # Feb 8 .. Feb 19
    vix = _vix(["2020-01-01"], [15.0])
    fred = pd.DataFrame({"date": pd.to_datetime(["2019-12-01", "2020-01-01"], utc=True),
                         "CPIAUCSL": [99.0, 100.0]})
    out = dl.merge_all_sources(bars, fred, vix).set_index("timestamp")["CPIAUCSL"]
    at = lambda s: out[pd.Timestamp(s, tz="UTC")]
    # Jan CPI is dated Jan 1; with the 45-day lag it becomes available at exactly 2020-02-15 00:00.
    assert at("2020-02-10 12:00") == 99.0
    assert at("2020-02-14 23:00") == 99.0          # one hour before: still the December print
    assert at("2020-02-15 00:00") == 100.0         # available from the instant obs_date + 45d
    assert at("2020-02-16 12:00") == 100.0


@pytest.mark.parametrize("series,lag", [("GDP", 120), ("UNRATE", 38), ("FEDFUNDS", 35)])
def test_default_lags_registered(series, lag):
    assert dl.DEFAULT_FRED_RELEASE_LAGS_DAYS[series] == lag


def test_gdp_not_visible_at_quarter_start():
    bars = _bars("2020-01-02", 24 * 130)                       # Jan 2 .. May 11
    vix = _vix(["2019-12-01"], [15.0])
    fred = pd.DataFrame({"date": pd.to_datetime(["2019-10-01", "2020-01-01"], utc=True),
                         "GDP": [21000.0, 21500.0]})
    out = dl.merge_all_sources(bars, fred, vix).set_index("timestamp")["GDP"]
    at = lambda s: out[pd.Timestamp(s, tz="UTC")]
    # Q4'19 (obs Oct 1) is available from Oct 1 + 120d = Jan 29; earlier rows are trimmed.
    assert out.index[0] == pd.Timestamp("2020-01-29", tz="UTC")
    assert at("2020-02-10 00:00") == 21000.0
    # Q1'20 (obs Jan 1) must NOT be visible in early April despite being "dated" Jan 1.
    assert at("2020-04-05 00:00") == 21000.0
    assert at("2020-04-29 23:00") == 21000.0
    assert at("2020-04-30 00:00") == 21500.0               # Jan 1 + 120d


def test_zero_lags_reproduces_legacy_same_day_join():
    bars = _bars("2020-03-09 00:00", 72)
    vix = _vix(["2020-03-09", "2020-03-10", "2020-03-11"], [10.0, 20.0, 30.0])
    fred = pd.DataFrame({"date": pd.to_datetime(["2020-03-01", "2020-03-10"], utc=True), "X": [1.0, 2.0]})
    out = dl.merge_all_sources(bars, fred, vix, **ZERO)
    legacy = bars.assign(date=bars["timestamp"].dt.floor("D")).merge(
        vix.rename(columns={"date": "date"}), on="date", how="left")
    assert np.array_equal(out["vix"].to_numpy(), legacy["vix"].ffill().to_numpy())
    assert out.loc[out["timestamp"] == "2020-03-10 05:00", "X"].item() == 2.0


def test_long_format_fred_is_pivoted(tmp_path):
    f = tmp_path / "fred.csv"
    f.write_text("date,series_id,value\n2020-01-01,CPIAUCSL,100\n2020-01-01,UNRATE,3.5\n"
                 "2020-02-01,CPIAUCSL,101\n2020-02-01,UNRATE,.\n")
    df = dl.load_fred(str(f))
    assert set(df.columns) == {"date", "CPIAUCSL", "UNRATE"}     # no 'series_id' / 'value' leakage
    assert df["CPIAUCSL"].tolist() == [100.0, 101.0]
    assert np.isnan(df["UNRATE"].iloc[1])                          # FRED's '.' -> NaN


def test_missing_marker_holds_last_valid_observation():
    bars = _bars("2020-06-01", 24 * 3)
    vix = _vix(["2020-05-01"], [15.0])
    fred = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-02-01"], utc=True),
                         "UNRATE": [3.5, np.nan]})
    out = dl.merge_all_sources(bars, fred, vix)
    assert (out["UNRATE"] == 3.5).all()


def test_column_collision_renamed_and_prices_untouched():
    bars = _bars("2020-03-10", 48)
    bars["volume"] = np.arange(48.0)
    vix = _vix(["2020-03-01"], [15.0])
    fred = pd.DataFrame({"date": pd.to_datetime(["2020-03-01"], utc=True), "vix": [99.0], "volume": [-1.0]})
    out = dl.merge_all_sources(bars, fred, vix, **ZERO)
    assert (out["vix"] == 15.0).all() and (out["vix_fred"] == 99.0).all()
    assert out["volume"].tolist() == list(np.arange(48.0))       # OHLCV column NOT overwritten
    assert (out["volume_fred"] == -1.0).all()


def test_leading_rows_trimmed_and_rest_preserved():
    bars = _bars("2020-03-09 00:00", 72)
    vix = _vix(["2020-03-09", "2020-03-10"], [10.0, 20.0])
    fred = pd.DataFrame({"date": pd.to_datetime(["2020-03-01"], utc=True), "X": [1.0]})
    out = dl.merge_all_sources(bars, fred, vix)                   # vix first usable 2020-03-10 00:00
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-03-10 00:00", tz="UTC")
    assert len(out) == 48 and not out.isna().any().any()
    assert out["timestamp"].is_monotonic_increasing
    kept = dl.merge_all_sources(bars, fred, vix, drop_leading_incomplete=False)
    assert len(kept) == 72 and kept["vix"].isna().sum() == 24


def test_only_expected_columns_added_no_helper_leak():
    bars = _bars("2020-03-10", 48)
    fred = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"], utc=True), "CPIAUCSL": [1.0]})
    out = dl.merge_all_sources(bars, fred, _vix(["2020-03-01"], [15.0]), **ZERO)
    assert list(out.columns) == list(bars.columns) + ["vix", "CPIAUCSL"]


# ---------------------------------------------------------------- OHLCV
def test_ohlcv_dedupes_drops_bad_rows_and_accepts_alias(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("Datetime,Open,High,Low,Close,Volume\n"
                 "2020-01-01 00:01:00,1.1,1.2,1.0,1.15,5\n"
                 "2020-01-01 00:00:00,1.1,1.2,1.0,1.10,5\n"
                 "2020-01-01 00:01:00,1.1,1.2,1.0,1.16,7\n"      # duplicate ts, later wins
                 + "".join(f"2020-01-01 {h:02d}:{m:02d}:00,1.1,1.2,1.0,1.1,1\n"
                           for h in (1, 2, 3, 4) for m in range(50))
                 + "2020-01-01 05:00:00,0,0,0,0,0\n")             # non-positive -> dropped (<1%)
    df = dl.load_dukascopy(str(f))
    assert df["timestamp"].is_monotonic_increasing and df["timestamp"].is_unique
    assert df.loc[df["timestamp"] == "2020-01-01 00:01:00", "close"].item() == 1.16
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


def test_ohlcv_mostly_bad_file_raises(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("timestamp,open,high,low,close,volume\n" + "garbage,x,y,z,w,v\n" * 20 + "2020-01-01,1,1,1,1,1\n")
    with pytest.raises(ValueError, match="invalid"):
        dl.load_dukascopy(str(f))


def test_ohlcv_missing_columns_named_in_error(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("timestamp,open,close\n2020-01-01,1,1\n")
    with pytest.raises(ValueError, match="high"):
        dl.load_dukascopy(str(f))


def test_unnamed_index_header_from_old_dukascopy_cache_is_accepted(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text(",open,high,low,close,volume\n2020-01-01 00:00:00+00:00,1,1,1,1,1\n")
    assert len(dl.load_dukascopy(str(f))) == 1


# ---------------------------------------------------------------- VIX / yfinance
class FakeYF:
    """Stands in for the yfinance module."""
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def download(self, ticker, **kw):
        self.calls.append((ticker, kw))
        r = self.responses.pop(0) if self.responses else self.responses_last
        if isinstance(r, Exception):
            raise r
        return r


def _yf_frame(start="2009-12-01", end="2025-12-31", multiindex=True):
    idx = pd.bdate_range(start, end)
    idx.name = "Date"
    vals = np.linspace(15, 25, len(idx))
    cols = {"Close": vals, "High": vals, "Low": vals, "Open": vals, "Volume": 0}
    df = pd.DataFrame(cols, index=idx)
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["^VIX"]], names=["Price", "Ticker"])
    return df


@pytest.fixture()
def no_sleep(monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)


def _install(monkeypatch, fake):
    monkeypatch.setattr(dl, "yf", fake)
    monkeypatch.setattr(dl, "_HAS_YFINANCE", True)


def test_end_date_is_inclusive(tmp_path, monkeypatch):
    fake = FakeYF([_yf_frame()])
    _install(monkeypatch, fake)
    dl.load_vix(cache_path=str(tmp_path / "v.csv"), start="2010-01-01", end="2025-12-31")
    assert fake.calls[0][1]["end"] == "2026-01-01"          # yfinance `end` is exclusive


def test_empty_download_is_retried_then_succeeds(tmp_path, monkeypatch, no_sleep):
    fake = FakeYF([pd.DataFrame(), RuntimeError("429 Too Many Requests"), _yf_frame()])
    _install(monkeypatch, fake)
    df = dl.load_vix(cache_path=str(tmp_path / "v.csv"), start="2010-01-01", end="2025-12-31")
    assert len(fake.calls) == 3 and len(df) > 4000
    assert (tmp_path / "v.csv").exists()


def test_failed_download_never_poisons_cache(tmp_path, monkeypatch, no_sleep):
    fake = FakeYF([pd.DataFrame()] * 3)
    _install(monkeypatch, fake)
    cache = tmp_path / "v.csv"
    with pytest.raises(dl.DataFileNotFoundError):
        dl.load_vix(cache_path=str(cache), start="2010-01-01", end="2025-12-31")
    assert not cache.exists()                                # old code wrote an empty cache here


def test_valid_cache_skips_network(tmp_path, monkeypatch):
    cache = tmp_path / "v.csv"
    dl._normalize_vix_frame(_yf_frame()).to_csv(cache, index=False)
    fake = FakeYF([RuntimeError("network must not be touched")])
    _install(monkeypatch, fake)
    df = dl.load_vix(cache_path=str(cache), start="2010-01-01", end="2025-12-31")
    assert fake.calls == [] and len(df) > 4000


def test_stale_partial_cache_is_rejected_and_refetched(tmp_path, monkeypatch):
    cache = tmp_path / "v.csv"
    dl._normalize_vix_frame(_yf_frame("2009-12-01", "2012-12-31")).to_csv(cache, index=False)
    fake = FakeYF([_yf_frame()])
    _install(monkeypatch, fake)
    df = dl.load_vix(cache_path=str(cache), start="2010-01-01", end="2025-12-31")
    assert len(fake.calls) == 1 and df["date"].max() >= pd.Timestamp("2025-12-24", tz="UTC")
    assert pd.read_csv(cache)["date"].max() >= "2025-12-24"  # cache repaired


def test_partial_download_not_accepted_or_cached(tmp_path, monkeypatch, no_sleep):
    fake = FakeYF([_yf_frame("2009-12-01", "2015-01-01")])
    _install(monkeypatch, fake)
    cache = tmp_path / "v.csv"
    with pytest.raises(dl.DataFileNotFoundError, match="does not cover"):
        dl.load_vix(cache_path=str(cache), start="2010-01-01", end="2025-12-31")
    assert not cache.exists()


def test_yfinance_multirow_header_csv_cache_is_parsed(tmp_path):
    body = _yf_frame(multiindex=False)
    raw = ["Price,Close,High,Low,Open,Volume", "Ticker,^VIX,^VIX,^VIX,^VIX,^VIX", "Date,,,,,"]
    raw += [f"{d.date()},{c},{c},{c},{c},0" for d, c in zip(body.index, body["Close"])]
    p = tmp_path / "v.csv"
    p.write_text("\n".join(raw))
    out = dl._normalize_vix_frame(pd.read_csv(p))
    assert len(out) == len(body) and out["vix"].dtype == "float64"     # junk header rows dropped


def test_tz_aware_dates_and_nonpositive_values_handled():
    raw = pd.DataFrame({"Date": pd.to_datetime(["2020-03-09 00:00-04:00", "2020-03-10 00:00-04:00",
                                                "2020-03-10 00:00-04:00", "2020-03-11 00:00-04:00"]),
                        "Close": [10.0, 0.0, 21.0, 30.0]})
    out = dl._normalize_vix_frame(raw)
    assert out["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-03-09", "2020-03-10", "2020-03-11"]
    assert out["vix"].tolist() == [10.0, 21.0, 30.0]                   # 0.0 dropped, dup date -> kept once


def test_cache_write_is_atomic(tmp_path):
    from src.utils.io import atomic_write_text
    p = tmp_path / "c.csv"
    atomic_write_text(p, "a\n1\n")
    assert p.read_text() == "a\n1\n" and [x.name for x in tmp_path.iterdir()] == ["c.csv"]


# ================================================================= acquire_all_data.py
@pytest.fixture()
def acq(tmp_path, monkeypatch):
    try:                                     # scripts/ at the repo root, or inside src/
        import scripts.acquire_all_data as a
    except ImportError:
        import src.scripts.acquire_all_data as a
    monkeypatch.setattr(a, "START_YEAR", 2010)
    monkeypatch.setattr(a, "END_YEAR", 2012)
    # EFFECTIVE_END_YEAR is computed once at import time from the real
    # config/real clock (see _effective_end_year) -- patching END_YEAR
    # alone doesn't retroactively update it, so it needs patching too,
    # same as FRED_END below already does for the same reason.
    monkeypatch.setattr(a, "EFFECTIVE_END_YEAR", 2012)
    monkeypatch.setattr(a, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(a, "DUKASCOPY_MASTER", str(tmp_path / "master.csv"))
    monkeypatch.setattr(a, "XVAL_FILE", str(tmp_path / "xval.csv"))
    monkeypatch.setattr(a, "FRED_FILE", str(tmp_path / "fred.csv"))
    monkeypatch.setattr(a, "FRED_START", "2009-01-01")
    monkeypatch.setattr(a, "FRED_END", "2012-12-31")
    monkeypatch.setattr(a.time, "sleep", lambda s: None)
    return a


def _year_frame(year, n=200_000, name="timestamp", start_offset_days=0):
    ts = pd.date_range(f"{year}-01-03", f"{year}-12-30 23:59", periods=n, tz="UTC").round("min")
    df = pd.DataFrame({"open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1, "volume": 1.0}, index=ts)
    df.index.name = name
    return df


def test_validate_year_accepts_good_and_rejects_truncated(acq):
    acq._validate_year(_year_frame(2011), 2011)
    with pytest.raises(ValueError, match="truncated"):
        acq._validate_year(_year_frame(2011).iloc[:100_000], 2011)         # ends mid-year
    with pytest.raises(ValueError, match="empty"):
        acq._validate_year(pd.DataFrame(), 2011)
    with pytest.raises(ValueError, match="missing columns"):
        acq._validate_year(_year_frame(2011).drop(columns=["close"]), 2011)


def test_consolidate_builds_master_with_timestamp_column_sorted_deduped(acq, tmp_path):
    os.makedirs(acq.CACHE_DIR)
    for y in (2010, 2011, 2012):
        df = _year_frame(y, n=50)
        if y == 2011:
            df.index.name = None                                    # old caches: blank index header
        df.to_csv(acq._year_cache(y))
    # overlapping boundary bar (last bar of 2010 repeated at start of 2011) + an out-of-window bar
    extra = pd.read_csv(acq._year_cache(2010), index_col=0).tail(1)
    pd.concat([extra, pd.read_csv(acq._year_cache(2011), index_col=0)]).to_csv(acq._year_cache(2011))
    late = _year_frame(2012, n=1); late.index = pd.DatetimeIndex(["2013-01-01 00:00:00+00:00"], name="timestamp")
    pd.concat([pd.read_csv(acq._year_cache(2012), index_col=0), late]).to_csv(acq._year_cache(2012))

    acq.consolidate_dukascopy()
    m = pd.read_csv(acq.DUKASCOPY_MASTER)
    assert m.columns[0] == "timestamp"
    ts = pd.to_datetime(m["timestamp"], utc=True)
    assert ts.is_monotonic_increasing and ts.is_unique
    assert len(m) == 150 and ts.max() < pd.Timestamp("2013-01-01", tz="UTC")
    assert len(dl.load_dukascopy(acq.DUKASCOPY_MASTER)) == 150         # the loader accepts what we produce


def test_consolidate_refuses_when_a_year_is_missing(acq):
    os.makedirs(acq.CACHE_DIR)
    _year_frame(2010, n=50).to_csv(acq._year_cache(2010))
    with pytest.raises(RuntimeError, match=r"missing year caches: \[2011, 2012\]"):
        acq.consolidate_dukascopy()


def test_current_incomplete_year_is_skipped_not_refused(acq, monkeypatch):
    """Behavior changed: acquire_dukascopy() used to abort outright
    (SystemExit) when END_YEAR named the current, not-yet-complete
    year. It now SKIPS that one year and proceeds with whatever IS
    complete (EFFECTIVE_END_YEAR) instead -- see _effective_end_year().
    This test overrides the `acq` fixture's own EFFECTIVE_END_YEAR=2012
    (set unconditionally for every other test using this fixture) to
    actually exercise that computation here specifically."""
    import datetime as dt
    now_year = dt.datetime.now(dt.timezone.utc).year
    monkeypatch.setattr(acq, "END_YEAR", now_year)
    monkeypatch.setattr(acq, "EFFECTIVE_END_YEAR", acq._effective_end_year(now_year))
    assert acq.EFFECTIVE_END_YEAR == now_year - 1
    # No SystemExit -- it proceeds (and fails for an unrelated, expected
    # reason: no real Dukascopy data source is reachable/installed in
    # this test environment, a RuntimeError, not the old SystemExit).
    with pytest.raises(RuntimeError):
        acq.acquire_dukascopy()


def test_histdata_timestamps_converted_from_est_to_utc(acq):
    raw = "20100104 170000;1.4;1.4;1.4;1.4;0\n20100104 170100;1.4;1.4;1.4;1.4;0\n"
    df = acq._parse_histdata_csv(io.StringIO(raw))
    assert str(df["timestamp"].iloc[0]) == "2010-01-04 22:00:00+00:00"   # 17:00 EST == 22:00 UTC
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_histdata_zip_with_readme_txt_reads_only_the_csv(acq, tmp_path, monkeypatch):
    zp = tmp_path / "HISTDATA_COM_ASCII_EURUSD_M12010.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("DAT_ASCII_EURUSD_M1_2010.csv", "20100104 170000;1;1;1;1;0\n")
        z.writestr("DAT_ASCII_EURUSD_M1_2010.txt", "this is a readme, not data")
    monkeypatch.setattr(acq, "HISTDATA_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "histdata", type(sys)("histdata"))
    monkeypatch.setitem(sys.modules, "histdata.api", type(sys)("histdata.api"))
    sys.modules["histdata"].download_hist_data = lambda **k: (_ for _ in ()).throw(AssertionError("no download"))
    sys.modules["histdata.api"].Platform = sys.modules["histdata.api"].TimeFrame = object
    assert len(acq._histdata_year(2010)) == 1


class _Resp:
    def __init__(self, text="", js=None, status=200):
        self.text, self._js, self.status_code = text, js, status
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def json(self):
        return self._js


def _fake_requests(monkeypatch, handler):
    import requests
    monkeypatch.setattr(requests, "get", handler)


def _fredgraph(sid, dates, vals, datecol="observation_date"):
    return _Resp(text=f"{datecol},{sid}\n" + "\n".join(f"{d},{v}" for d, v in zip(dates, vals)) + "\n")


def test_fred_keyless_pulls_all_series_writes_wide_file(acq, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(acq, "FRED_SERIES", ["CPIAUCSL", "GDP"])
    def handler(url, params=None, timeout=None):
        assert "api_key" not in (params or {}) and timeout is not None
        d = pd.date_range("2009-01-01", "2012-12-01", freq="MS").strftime("%Y-%m-%d")
        v = ["."] + [str(100 + i) for i in range(1, len(d))]           # FRED marks missing as '.'
        return _fredgraph(params["id"], d, v, datecol="DATE" if params["id"] == "GDP" else "observation_date")
    _fake_requests(monkeypatch, handler)
    acq.acquire_fred()
    out = dl.load_fred(acq.FRED_FILE)
    assert set(out.columns) == {"date", "CPIAUCSL", "GDP"}
    assert out["date"].min() == pd.Timestamp("2009-02-01", tz="UTC")     # '.' row dropped, lookback kept


def test_fred_official_api_used_when_key_set_and_key_never_printed(acq, monkeypatch, capsys):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123")
    monkeypatch.setattr(acq, "FRED_SERIES", ["UNRATE"])
    def handler(url, params=None, timeout=None):
        assert "api.stlouisfed.org" in url and params["api_key"] == "SECRETKEY123"
        obs = [{"date": d.strftime("%Y-%m-%d"), "value": "3.5"} for d in pd.date_range("2009-01-01", "2012-12-01", freq="MS")]
        return _Resp(js={"observations": obs})
    _fake_requests(monkeypatch, handler)
    acq.acquire_fred()
    assert "SECRETKEY123" not in capsys.readouterr().out


def test_fred_stale_series_is_rejected_and_nothing_written(acq, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(acq, "FRED_SERIES", ["UNRATE"])
    _fake_requests(monkeypatch, lambda url, params=None, timeout=None:
                   _fredgraph("UNRATE", ["2009-01-01", "2010-01-01"], [5, 6]))
    with pytest.raises(RuntimeError, match="too old"):
        acq.acquire_fred()
    assert not os.path.exists(acq.FRED_FILE)


def test_fred_one_failing_series_writes_no_partial_file(acq, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(acq, "FRED_SERIES", ["UNRATE", "GDP"])
    def handler(url, params=None, timeout=None):
        if params["id"] == "GDP":
            return _Resp(status_code=503)
        d = pd.date_range("2009-01-01", "2012-12-01", freq="MS").strftime("%Y-%m-%d")
        return _fredgraph("UNRATE", d, [4.0] * len(d))
    _fake_requests(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="FRED GDP"):
        acq.acquire_fred()
    assert not os.path.exists(acq.FRED_FILE)


def test_old_fred_file_without_lookback_or_series_triggers_refetch(acq, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(acq, "FRED_SERIES", ["UNRATE"])
    pd.DataFrame({"date": ["2010-01-01", "2010-02-01"], "UNRATE": [1, 2]}).to_csv(acq.FRED_FILE, index=False)
    calls = []
    def handler(url, params=None, timeout=None):
        calls.append(params["id"])
        d = pd.date_range("2009-01-01", "2012-12-01", freq="MS").strftime("%Y-%m-%d")
        return _fredgraph("UNRATE", d, [4.0] * len(d))
    _fake_requests(monkeypatch, handler)
    acq.acquire_fred()
    assert calls == ["UNRATE"] and pd.read_csv(acq.FRED_FILE)["date"].min() == "2009-01-01"


def test_csv_summary_counts_rows_without_loading(acq, tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame({"timestamp": ["2010-01-03 22:00:00+00:00", "2010-01-03 22:01:00+00:00",
                                "2025-12-30 21:59:00+00:00"], "close": [1, 2, 3]}).to_csv(p, index=False)
    n, first, last, header = acq._csv_summary(str(p))
    assert (n, first, last, header) == (3, "2010-01-03 22:00:00+00:00", "2025-12-30 21:59:00+00:00", "timestamp,close")


# ---------------------------------------------------------------- temporal_split
def test_temporal_split_end_dates_are_inclusive_and_splits_are_exhaustive():
    ts = pd.date_range("2021-12-30", "2022-01-02 23:59", freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "x": range(len(ts))})
    cfg = dict(train_start="2021-12-30", train_end="2021-12-31",
               val_start="2022-01-01", val_end="2022-01-01",
               test_start="2022-01-02", test_end="2022-01-02")
    sp = dl.temporal_split(df, cfg)
    assert sp["train"]["timestamp"].max() == pd.Timestamp("2021-12-31 23:00", tz="UTC")   # was 00:00
    assert sum(len(v) for v in sp.values()) == len(df)                                    # no bar lost
    assert len(set(sp["train"].index) & set(sp["val"].index)) == 0                         # no overlap
