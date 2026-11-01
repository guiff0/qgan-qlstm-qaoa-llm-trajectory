"""Dukascopy direct client: throttling, 429 back-off, per-day cache, circuit breaker.
No network: requests.Session is replaced by a scripted fake."""
import datetime as dt
import lzma
import os
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QGAN_BASE_DIR", os.getcwd())
sys.path.insert(0, os.getcwd())

from scripts import acquire_all_data as a

REC = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])


def _body(n=3, price=110000):
    arr = np.zeros(n, dtype=REC)
    arr["t"] = np.arange(n) * 60
    arr["o"] = arr["c"] = arr["l"] = arr["h"] = price
    arr["v"] = 1.0
    return lzma.compress(arr.tobytes())


class _Resp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


class _FakeSession:
    """script(url, n_call_for_this_url) -> _Resp"""
    calls = []
    script = staticmethod(lambda url, n: _Resp(200, _body()))

    def __init__(self):
        self.headers = {}

    def get(self, url, timeout=None):
        n = sum(1 for u in _FakeSession.calls if u == url)
        _FakeSession.calls.append(url)
        return _FakeSession.script(url, n)


@pytest.fixture
def fast(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests, "Session", _FakeSession)
    _FakeSession.calls = []
    monkeypatch.setattr(a, "DAY_CACHE_DIR", str(tmp_path / "days"))
    monkeypatch.setattr(a, "DUKA_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(a, "DUKA_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(a, "DUKA_BACKOFF_MAX", 0.02)
    monkeypatch.setattr(a, "DUKA_WORKERS", 2)
    return tmp_path


def _span(d0, d1):
    u = dt.timezone.utc
    return dt.datetime.fromisoformat(d0).replace(tzinfo=u), dt.datetime.fromisoformat(d1).replace(tzinfo=u)


def test_429_then_success_is_retried_and_cached(fast):
    _FakeSession.script = staticmethod(
        lambda url, n: _Resp(429, headers={"Retry-After": "0.01"}) if n < 2 else _Resp(200, _body()))
    df = a._fetch_dukascopy_direct(*_span("2024-03-04", "2024-03-06"))       # Mon, Tue
    assert len(df) == 6
    assert len(_FakeSession.calls) == 6                                      # 2 days x (2 x 429 + 1 x 200)
    _FakeSession.calls = []
    df2 = a._fetch_dukascopy_direct(*_span("2024-03-04", "2024-03-06"))
    assert _FakeSession.calls == [] and len(df2) == 6                        # resumed purely from the day cache


def test_saturdays_are_not_requested_and_404_days_are_remembered(fast):
    _FakeSession.script = staticmethod(lambda url, n: _Resp(404) if "/05/" in url else _Resp(200, _body()))
    # 2024-03-09 is a Saturday; 2024-03-06 -> URL month 02 (zero based), day 06
    a._fetch_dukascopy_direct(*_span("2024-03-08", "2024-03-11"))             # Fri, Sat, Sun
    assert not any(u.endswith("/2024/02/09/BID_candles_min_1.bi5") for u in _FakeSession.calls)
    assert len(_FakeSession.calls) == 2
    _FakeSession.calls = []
    a._fetch_dukascopy_direct(*_span("2024-03-08", "2024-03-11"))
    assert _FakeSession.calls == []


def test_404_is_cached_as_no_data(fast):
    _FakeSession.script = staticmethod(lambda url, n: _Resp(404))
    df = a._fetch_dukascopy_direct(*_span("2024-01-01", "2024-01-02"))        # Monday holiday
    assert df.empty
    _FakeSession.calls = []
    a._fetch_dukascopy_direct(*_span("2024-01-01", "2024-01-02"))
    assert _FakeSession.calls == []


def test_persistent_429_trips_the_breaker_early_and_keeps_good_days(fast, monkeypatch):
    monkeypatch.setattr(a, "DUKA_DAY_ATTEMPTS", 2)
    monkeypatch.setattr(a, "DUKA_BREAKER", 3)
    monkeypatch.setattr(a, "DUKA_WORKERS", 1)
    # first weekday succeeds, everything after is refused
    _FakeSession.script = staticmethod(
        lambda url, n: _Resp(200, _body()) if "/2024/00/01/" in url else _Resp(429))
    with pytest.raises(a.DukascopyRateLimited, match="re-run"):
        a._fetch_dukascopy_direct(*_span("2024-01-01", "2024-03-01"))        # ~43 weekdays
    assert len(_FakeSession.calls) < 15                                       # stopped early, not 43 days x attempts
    assert os.path.isfile(a._day_paths(pd.Timestamp("2024-01-01"))[0])        # the good day survived


def test_corrupt_body_is_not_cached(fast, monkeypatch):
    monkeypatch.setattr(a, "DUKA_DAY_ATTEMPTS", 1)
    _FakeSession.script = staticmethod(lambda url, n: _Resp(200, b"not lzma"))
    with pytest.raises(Exception):
        a._fetch_dukascopy_direct(*_span("2024-03-04", "2024-03-05"))
    assert not os.path.isfile(a._day_paths(pd.Timestamp("2024-03-04"))[0])


def test_rate_limit_is_fatal_in_retry_wrapper():
    n = {"c": 0}

    def boom():
        n["c"] += 1
        raise a.DukascopyRateLimited("x")
    with pytest.raises(a.DukascopyRateLimited):
        a._retry(boom, "t", attempts=3, base_sleep=0, fatal=(a.DukascopyRateLimited,))
    assert n["c"] == 1


def test_persistent_5xx_on_one_day_is_skipped_not_fatal(fast, monkeypatch):
    monkeypatch.setattr(a, "DUKA_DAY_ATTEMPTS", 2)
    monkeypatch.setattr(a, "GAP_LOG", str(fast / "missing_days.csv"))
    _FakeSession.script = staticmethod(
        lambda url, n: _Resp(503) if "/2024/02/05/" in url else _Resp(200, _body()))   # 2024-03-05
    df = a._fetch_dukascopy_direct(*_span("2024-03-04", "2024-03-07"))               # Mon-Wed
    assert len(df) == 6                                                              # 2 good days kept
    assert "2024-03-05" in open(a.GAP_LOG).read()
    assert not os.path.isfile(a._day_paths(pd.Timestamp("2024-03-05"))[1])           # not cached as 'no data'


def test_5xx_everywhere_is_still_fatal(fast, monkeypatch):
    monkeypatch.setattr(a, "DUKA_DAY_ATTEMPTS", 1)
    monkeypatch.setattr(a, "DUKA_MAX_GAP_DAYS", 2)
    monkeypatch.setattr(a, "GAP_LOG", str(fast / "missing_days.csv"))
    _FakeSession.script = staticmethod(lambda url, n: _Resp(503))
    with pytest.raises(a.DukascopyRateLimited, match="unhealthy"):
        a._fetch_dukascopy_direct(*_span("2024-03-04", "2024-03-12"))


def test_429_on_a_day_is_not_downgraded_to_a_gap(fast, monkeypatch):
    monkeypatch.setattr(a, "DUKA_DAY_ATTEMPTS", 2)
    _FakeSession.script = staticmethod(lambda url, n: _Resp(503) if n == 0 else _Resp(429))
    with pytest.raises(a.DukascopyRateLimited):
        a._fetch_dukascopy_direct(*_span("2024-03-04", "2024-03-05"))
