"""
Acquire raw data from the API sources into data/raw/.

    python -m scripts.acquire_all_data                     # everything
    python -m scripts.acquire_all_data --steps fred        # just FRED
    python -m scripts.acquire_all_data --refresh-fred      # re-pull FRED (e.g. to add the lookback)

Output paths come from config/default_config.yaml (data.dukascopy_file,
data.forexsb_file, data.fred_file) so this script and the loaders in
src/data/data_loader.py can never disagree about where files live. The study
window comes from data.train_start / data.test_end.

FIXES vs. the previous version
  Dukascopy
    1. NO CONSOLIDATION STEP EXISTED. Year files were written to a cache dir but
       nothing ever produced the master CSV that verify_pipeline() and the
       loaders expect (the "Consolidator & Deduplicator" section was empty).
    2. Year cache files were written with an unnamed index, so they had no
       `timestamp` column. The index is now named.
    3. Writes were not atomic; a crash mid-write left a truncated file that the
       "exists and size > 0" cache check trusted forever. Now temp file +
       os.replace, and a fetched year is validated (starts near Jan 1, ends near
       Dec 31) before it is allowed into the cache.
    4. Failed years were printed and skipped, silently leaving holes in a 16-year
       series. Failures are retried with backoff and then reported and fatal.
    5. The current (incomplete) year was fetched and cached as if complete, and
       FRED's end date used the current year rather than the study window.
       The window is now fixed by config, and an incomplete year is refused.
  Cross-validation (HistData.com; the old code called it "ForexSB")
    6. Only year 2010 was ever downloaded, into a file named ..._2010_2023.csv.
       Every year in the window is now fetched.
    7. Timestamps were left timezone-naive. HistData ASCII is fixed EST (UTC-5,
       no DST), so the downstream loader (which assumes UTC) was 5 hours off.
    8. Only the first CSV found was read; a dead `download_hist_data`-style
       function with unreachable mirrors, and `urlopen` calls with no timeout,
       are gone.
  FRED
    9. End date was the current year, not the study window; start was the study
       start, so the first months of 2010 had no *previously published* value
       to apply (the loader now applies publication lags). Fetches from one
       year earlier.
   10. pandas_datareader (unmaintained, breaks on recent pandas) replaced by the
       official FRED API when FRED_API_KEY is set, else FRED's keyless CSV
       endpoint. Requests have timeouts, retries and validation; the file is
       only written if every series came back with recent data.
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import time
import zipfile
import datetime as dt

import pandas as pd

def _find_repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isfile(os.path.join(here, "config", "default_config.yaml")):
            return here
        here = os.path.dirname(here)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REPO_ROOT = os.environ.get("QGAN_BASE_DIR") or _find_repo_root()
sys.path.insert(0, REPO_ROOT)

from src.utils.config import load_config  # noqa: E402
from src.utils.progress import progress_bar  # noqa: E402

CFG = load_config()["data"]
DUKASCOPY_MASTER = CFG["dukascopy_file"]
XVAL_FILE = CFG["forexsb_file"]
FRED_FILE = CFG["fred_file"]
RAW_DIR = os.path.dirname(DUKASCOPY_MASTER) or "."
CACHE_DIR = os.path.join(RAW_DIR, "dukascopy_cache")
HISTDATA_DIR = os.path.join(RAW_DIR, "histdata_zips")

def _effective_end_year(configured_end_year: int, now: dt.datetime = None) -> int:
    """
    The configured end year (data.test_end), capped at the last year
    that has ACTUALLY finished. If test_end names a year that hasn't
    closed yet (e.g. set to 2026-12-31 while it's still September
    2026), this returns the last complete year instead (2025) and
    prints a clear note -- rather than the old behavior of aborting the
    whole acquisition step outright. A run today fetches/consolidates
    through the last complete year only; a run any time after the
    configured year actually closes picks it up automatically into the
    SAME master file, with no code or config change either way. (See
    _refuse_if_incomplete_year below for a stricter, raise-instead-of-
    skip alternative, still available but no longer the default here.)
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    if configured_end_year >= now.year:
        effective = now.year - 1
        print(f"[NOTE] data.test_end names {configured_end_year}, which is not a complete "
              f"year yet (today is {now.date()}); using {effective} for this run instead. "
              f"No code change needed -- re-run after {configured_end_year} closes to pick "
              f"it up automatically.")
        return effective
    return configured_end_year


START_YEAR = pd.Timestamp(CFG["train_start"]).year
END_YEAR = pd.Timestamp(CFG["test_end"]).year          # the configured/intended study window
EFFECTIVE_END_YEAR = _effective_end_year(END_YEAR)      # what this run can ACTUALLY fetch today
XVAL_LAST_YEAR = min(2023, EFFECTIVE_END_YEAR)  # cross-check window per Appendix C
FRED_SERIES = list(CFG.get("fred_series", ["CPIAUCSL", "FEDFUNDS", "GS10", "UNRATE", "GDP"]))
FRED_START = f"{START_YEAR - 1}-01-01"  # one year of lookback for publication lags
FRED_END = f"{EFFECTIVE_END_YEAR}-12-31"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _atomic_to_csv(df: pd.DataFrame, path: str, **kw) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        df.to_csv(tmp, **kw)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _retry(fn, what: str, attempts: int = 3, base_sleep: float = 5.0, fatal: tuple = ()):
    """Retry `fn` with exponential backoff. Exceptions in `fatal` are re-raised at once: retrying a
    rate-limited provider just burns time and makes the ban longer."""
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except fatal:
            raise
        except Exception as e:  # noqa: BLE001 - provider libs raise anything
            last = e
            print(f"    [RETRY {i}/{attempts}] {what}: {type(e).__name__}: {e}")
            if i < attempts:
                time.sleep(base_sleep * 2 ** (i - 1))
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}") from last


def _nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


# ----------------------------------------------------------------------
# [1/3] Dukascopy (dukascopy_python), one file per year, then consolidate
# ----------------------------------------------------------------------
def _year_cache(year: int) -> str:
    return os.path.join(CACHE_DIR, f"eurusd_1min_{year}.csv")


# Evidence-based, not a guess: this project's own prior successful runs
# have logged 2012-01-11 01:37:00+00:00 as the real first Dukascopy
# EURUSD 1-min timestamp -- the free export's history simply doesn't go
# back to Jan 1 for this year. Add an entry here only when there's
# similarly concrete evidence (an actual successful fetch) of a
# provider's real earliest-available date for a given year.
KNOWN_PROVIDER_HISTORY_START = {
    2012: pd.Timestamp("2012-01-11", tz="UTC"),
}


def _expected_start_threshold(year: int) -> pd.Timestamp:
    """Latest first-timestamp we'll accept for `year` before calling it a
    truncated download -- the provider's real earliest date (if known)
    plus the same week of weekend/holiday slack used everywhere else,
    instead of unconditionally assuming every year has full-year history."""
    base = KNOWN_PROVIDER_HISTORY_START.get(year, pd.Timestamp(year, 1, 1, tz="UTC"))
    return base + pd.Timedelta(days=7)


def _validate_year(df: pd.DataFrame, year: int) -> None:
    if df is None or df.empty:
        raise ValueError("empty result")
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}; got {list(df.columns)}")
    idx = pd.to_datetime(df.index, utc=True)
    first, last = idx.min(), idx.max()
    # FX is closed on weekends/holidays, so allow a week of slack at each edge
    # (and, for a year with a known real provider-history start, slack past
    # THAT date instead of Jan 1 -- see KNOWN_PROVIDER_HISTORY_START above).
    start_threshold = _expected_start_threshold(year)
    if first > start_threshold or last < pd.Timestamp(year, 12, 24, tz="UTC"):
        raise ValueError(f"truncated download: spans {first} -> {last} "
                         f"(expected to start on/before {start_threshold.date()})")
    min_rows = 100_000
    if year in KNOWN_PROVIDER_HISTORY_START:
        # A genuinely partial first year should have proportionally fewer
        # rows, not the full-year minimum -- otherwise this check would
        # itself reject the same real, complete-as-it-gets year.
        days_covered = (pd.Timestamp(year, 12, 31, tz="UTC") - KNOWN_PROVIDER_HISTORY_START[year]).days + 1
        min_rows = int(100_000 * days_covered / 365)
    if len(df) < min_rows:
        raise ValueError(f"only {len(df):,} rows (expected >= {min_rows:,})")


# ---- direct Dukascopy datafeed client -------------------------------------------------------
# The first version fired one un-throttled request per calendar day from 8 threads (~5,100 requests
# for 2012-2025) and kept no per-day cache. Dukascopy answered HTTP 429 to nearly all of them, every
# year then "failed after 3 attempts" (25 minutes), and nothing downloaded was kept for the next run.
# Now: one shared throttle, back-off that honours Retry-After and pauses ALL workers, a per-day cache
# (a re-run only asks for days that are still missing), and a circuit breaker that stops the whole step
# early when the server keeps refusing instead of grinding through every remaining year.
DUKA_WORKERS = int(os.environ.get("DUKASCOPY_WORKERS", "2"))
DUKA_MIN_INTERVAL = float(os.environ.get("DUKASCOPY_MIN_INTERVAL", "0.35"))   # s between requests, all threads
DUKA_DAY_ATTEMPTS = int(os.environ.get("DUKASCOPY_DAY_ATTEMPTS", "6"))
DUKA_BACKOFF_BASE = float(os.environ.get("DUKASCOPY_BACKOFF_BASE", "10"))     # s; doubles per attempt
DUKA_BACKOFF_MAX = float(os.environ.get("DUKASCOPY_BACKOFF_MAX", "300"))
DUKA_BREAKER = int(os.environ.get("DUKASCOPY_BREAKER", "4"))                  # consecutive exhausted days
DAY_CACHE_DIR = os.path.join(CACHE_DIR, "days")


class DukascopyRateLimited(RuntimeError):
    """The server keeps answering 429/5xx/timeouts after back-off. Already-downloaded days are cached;
    wait (minutes to an hour) and re-run the same command."""


class _Throttle:
    """Shared by every worker thread: a minimum spacing between requests, plus a global cool-down
    (`penalize`) so a 429 seen by one thread pauses all of them, not just itself."""

    def __init__(self, min_interval: float):
        import threading
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_ok)
            self._next_ok = slot + self.min_interval
        if slot > now:
            time.sleep(slot - now)

    def penalize(self, seconds: float) -> None:
        with self._lock:
            self._next_ok = max(self._next_ok, time.monotonic() + seconds)


def _day_paths(day: pd.Timestamp) -> tuple[str, str]:
    stem = os.path.join(DAY_CACHE_DIR, f"EURUSD_{day:%Y%m%d}")
    return stem + ".bi5", stem + ".none"      # raw body / "server confirmed: no data that day"


def _parse_bi5(body: bytes, day: pd.Timestamp):
    """One day of 24-byte big-endian records (seconds-from-midnight, open, close, low, high : uint32 x 1e-5;
    volume : float32), LZMA-compressed. None when the day holds no candles."""
    import lzma
    import numpy as np
    if not body:
        return None
    rec = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"),
                    ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])
    arr = np.frombuffer(lzma.decompress(body), dtype=rec)
    if arr.size == 0:
        return None
    base = pd.Timestamp(year=day.year, month=day.month, day=day.day, tz="UTC")
    return pd.DataFrame({
        "open": arr["o"] / 1e5, "high": arr["h"] / 1e5,
        "low": arr["l"] / 1e5, "close": arr["c"] / 1e5,
        "volume": arr["v"].astype("float64"),
    }, index=base + pd.to_timedelta(arr["t"].astype("int64"), unit="s"))


def _write_bytes_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}_{id(data) & 0xffff}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _download_day(session, day: pd.Timestamp, throttle: _Throttle, stats: dict):
    """Body for one day, from the day cache when present, else from the server with back-off.
    Returns bytes, or None when the day has no data. Raises on a day that never succeeded."""
    import requests
    bi5, none = _day_paths(day)
    if os.path.isfile(none):
        stats["cached"] += 1
        return None
    if _nonempty(bi5):
        stats["cached"] += 1
        with open(bi5, "rb") as f:
            return f.read()
    # Dukascopy months in the URL are ZERO-based (January = 00).
    url = (f"https://datafeed.dukascopy.com/datafeed/EURUSD/"
           f"{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5")
    last = None
    for attempt in range(1, DUKA_DAY_ATTEMPTS + 1):
        throttle.wait()
        wait_s = None
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                _write_bytes_atomic(none, b"")
                stats["downloaded"] += 1
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                try:
                    wait_s = float(resp.headers.get("Retry-After", ""))
                except ValueError:
                    wait_s = None
                last = f"HTTP {resp.status_code}"
            else:
                resp.raise_for_status()
                body = resp.content
                stats["downloaded"] += 1
                if not body:
                    _write_bytes_atomic(none, b"")
                    return None
                _parse_bi5(body, day)          # corrupt body -> exception here, not at consolidation
                _write_bytes_atomic(bi5, body)
                return body
        except requests.RequestException as e:
            last = f"{type(e).__name__}"
        # exponential back-off with jitter, applied to ALL workers; the server's Retry-After wins
        import random
        pause = wait_s if wait_s is not None else min(DUKA_BACKOFF_BASE * 2 ** (attempt - 1), DUKA_BACKOFF_MAX)
        pause *= 1.0 + 0.2 * random.random()
        if attempt < DUKA_DAY_ATTEMPTS:
            throttle.penalize(pause)
    raise DukascopyRateLimited(f"{day.date()}: {last} after {DUKA_DAY_ATTEMPTS} attempts")


def _fetch_dukascopy_direct(start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Download EURUSD 1-min BID candles straight from Dukascopy's public datafeed, one .bi5 file per
    UTC day (see _parse_bi5 for the format). Used because the `dukascopy_python` package is broken
    (NameError: name 'requests' is not defined). Saturdays hold no FX data and are not requested;
    404 / empty days are remembered as such. Resumable: every day is cached as it arrives."""
    import threading
    import requests
    from concurrent.futures import ThreadPoolExecutor

    days = [d for d in pd.date_range(start.date(), (end - dt.timedelta(days=1)).date(), freq="D")
            if d.weekday() != 5]
    throttle = _Throttle(DUKA_MIN_INTERVAL)
    stats = {"cached": 0, "downloaded": 0}
    abort = threading.Event()
    fail_lock = threading.Lock()
    streak = {"n": 0, "err": None}

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research data acquisition; contact via repo owner)"})

    def one_day(day):
        if abort.is_set():
            return None
        try:
            body = _download_day(session, day, throttle, stats)
        except DukascopyRateLimited as e:
            with fail_lock:
                streak["n"] += 1
                streak["err"] = e
                if streak["n"] >= DUKA_BREAKER:
                    abort.set()
            raise
        with fail_lock:
            streak["n"] = 0
        return None if body is None else _parse_bi5(body, day)

    parts, errors = [], []
    with ThreadPoolExecutor(max_workers=max(1, DUKA_WORKERS)) as ex:
        for fut in [ex.submit(one_day, d) for d in days]:
            try:
                r = fut.result()
                if r is not None:
                    parts.append(r)
            except DukascopyRateLimited as e:
                errors.append(e)
            except Exception:           # corrupt body etc.: stop queuing work, surface the real error
                abort.set()
                raise
    if abort.is_set():
        raise DukascopyRateLimited(
            f"Dukascopy keeps refusing requests ({streak['err']}); stopped after {DUKA_BREAKER} consecutive "
            f"failed days. {stats['downloaded']} day(s) downloaded this pass are saved in {DAY_CACHE_DIR}. "
            f"Wait 15-60 min (the limit is per IP), then re-run the same command; it resumes. "
            f"Slower is safer: set DUKASCOPY_WORKERS=1 DUKASCOPY_MIN_INTERVAL=1.0")
    if errors:
        raise DukascopyRateLimited(f"{len(errors)} day(s) failed (first: {errors[0]}); re-run to retry only those")
    print(f"    days: {stats['cached']} from cache, {stats['downloaded']} downloaded")
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(parts).sort_index()


_PKG_WARNED = False


def fetch_dukascopy_year(year: int) -> None:
    global _PKG_WARNED
    cache = _year_cache(year)
    if _nonempty(cache):
        print(f" -> [CACHED] {year}")
        return

    dukascopy_python = None
    try:
        import dukascopy_python
        from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD
    except Exception as e:  # noqa: BLE001 - a broken package can raise NameError etc.
        if not _PKG_WARNED:       # once per run, not once per year
            print(f"    [WARN] dukascopy_python unusable ({type(e).__name__}: {e}); "
                  f"using the direct Dukascopy datafeed client instead")
            _PKG_WARNED = True
        dukascopy_python = None

    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)
    print(f" -> Fetching EURUSD 1-min bid for {year} ...")

    def _do():
        df = None
        if dukascopy_python is not None:
            try:
                df = dukascopy_python.fetch(
                    instrument=INSTRUMENT_FX_MAJORS_EUR_USD,
                    interval=dukascopy_python.INTERVAL_MIN_1,
                    offer_side=dukascopy_python.OFFER_SIDE_BID,
                    start=start, end=end,
                )
            except NameError as e:  # the package's own bug; not retryable
                print(f"    [WARN] dukascopy_python.fetch crashed ({e}); using direct client")
                df = None
        if df is None:
            df = _fetch_dukascopy_direct(start, end)
        df.columns = [str(c).lower().strip() for c in df.columns]
        _validate_year(df, year)
        return df

    df = _retry(_do, f"Dukascopy {year}", fatal=(DukascopyRateLimited,))
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    _atomic_to_csv(df, cache)  # only validated years ever reach the cache
    print(f"    [OK] {year}: {len(df):,} rows")


def consolidate_dukascopy() -> None:
    files = {y: _year_cache(y) for y in range(START_YEAR, EFFECTIVE_END_YEAR + 1)}
    missing = [y for y, f in files.items() if not _nonempty(f)]
    if missing:
        raise RuntimeError(f"Cannot consolidate; missing year caches: {missing}")

    if _nonempty(DUKASCOPY_MASTER) and \
            os.path.getmtime(DUKASCOPY_MASTER) >= max(os.path.getmtime(f) for f in files.values()):
        print(f" -> Master up to date: {DUKASCOPY_MASTER}")
        return

    print(f" -> Consolidating {len(files)} yearly files -> {DUKASCOPY_MASTER}")
    frames = []
    for y, f in files.items():
        d = pd.read_csv(f, index_col=0)  # tolerates old caches written with an unnamed index
        # ISO8601 accepts rows with/without fractional seconds; the default would infer ONE
        # format from the first row and raise on any row that differs.
        d.index = pd.to_datetime(d.index, utc=True, format="ISO8601")
        d.index.name = "timestamp"
        frames.append(d.reset_index())
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).lower().strip() for c in df.columns]
    df = df.sort_values("timestamp", kind="stable")
    n_dup = int(df["timestamp"].duplicated(keep="last").sum())
    df = df.drop_duplicates("timestamp", keep="last")
    lo = pd.Timestamp(START_YEAR, 1, 1, tz="UTC")
    hi = pd.Timestamp(EFFECTIVE_END_YEAR + 1, 1, 1, tz="UTC")
    df = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)].reset_index(drop=True)

    gaps = df["timestamp"].diff().dt.total_seconds() / 86400
    print(f"    rows={len(df):,} dropped_duplicates={n_dup:,} "
          f"span={df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]} "
          f"largest_gap={gaps.max():.1f}d")
    if gaps.max() > 6:
        print(f" -> [WARN] a gap of {gaps.max():.1f} days ending "
              f"{df.loc[gaps.idxmax(), 'timestamp']} -- inspect before training")
    _atomic_to_csv(df, DUKASCOPY_MASTER, index=False)
    print(f"    [OK] wrote {DUKASCOPY_MASTER}")


def _refuse_if_incomplete_year(end_year: int, now: dt.datetime = None) -> None:
    """NOT called by acquire_dukascopy()/acquire_fred() any more -- see
    _effective_end_year() above, which SKIPS an incomplete trailing year
    instead of aborting the whole acquisition step. This stricter,
    raise-instead-of-skip version is kept available (and still tested)
    for anyone who explicitly wants a hard failure instead -- e.g. a
    CI/scheduled job that should treat "2026 isn't done yet" as a real
    error rather than something to quietly work around.

    Refuses to fetch/cache a year that hasn't finished yet, rather than
    silently caching a partial year as if it were complete (see fix #5 in
    this file's module docstring). `now` is injectable for testing --
    without it, this check is only ever true or false depending on which
    day the test happens to run, which is exactly the kind of thing that
    should be a parameter, not a hidden call to dt.datetime.now()."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if end_year >= now.year:
        raise SystemExit(
            f"Study window ends in {end_year}, which is not a complete year yet "
            f"(today is {now.date()}); refusing to cache partial data. Re-run once "
            f"{end_year} has fully closed -- no code change needed, this check "
            f"clears itself automatically once the date passes. Set data.test_end "
            f"to an earlier, finished year if you need to run before then."
        )


def acquire_dukascopy() -> None:
    print(f"\n=== [1/3] Dukascopy EURUSD 1-min ({START_YEAR}-{EFFECTIVE_END_YEAR}"
          f"{f', study window extends to {END_YEAR}' if EFFECTIVE_END_YEAR != END_YEAR else ''}) ===")

    # Short-circuit if the master file already covers the study window --
    # matches the pattern acquire_fred() and acquire_crosscheck() already
    # use (both check their own output file before doing any work). Without
    # this, a master file that already exists (restored from backup, or a
    # previous run whose per-year data/raw/dukascopy_cache/ files were later
    # cleaned up) was ignored: every year was re-fetched from scratch, which
    # additionally required dukascopy_python to be installed even when
    # nothing actually needed fetching.
    if _nonempty(DUKASCOPY_MASTER):
        n, first, last, _ = _csv_summary(DUKASCOPY_MASTER)
        first_ts, last_ts = pd.Timestamp(first, tz="UTC"), pd.Timestamp(last, tz="UTC")
        first_ok = first_ts <= _expected_start_threshold(START_YEAR)
        last_ok = last_ts >= pd.Timestamp(EFFECTIVE_END_YEAR, 12, 24, tz="UTC")
        if first_ok and last_ok and n > 100_000:
            print(f" -> Master already covers {START_YEAR}-{EFFECTIVE_END_YEAR} ({n:,} rows, "
                  f"{first} -> {last}): {DUKASCOPY_MASTER}")
            return
        print(f" -> Master exists but does not cover the full window ({n:,} rows, "
              f"{first} -> {last}); re-fetching missing years.")
    failed = []
    years = list(range(START_YEAR, EFFECTIVE_END_YEAR + 1))
    bar = progress_bar(total=len(years), desc="Dukascopy years", unit="year")
    for y in years:
        try:
            fetch_dukascopy_year(y)
        except DukascopyRateLimited as e:
            # The server is refusing us: every remaining year would fail the same way, slowly.
            bar.close()
            raise RuntimeError(f"Dukascopy rate-limited at {y}: {e}") from e
        except Exception as e:  # noqa: BLE001
            print(f"    [ERROR] {y}: {e}")
            failed.append(y)
        bar.update(1)
        bar.set_postfix(year=y, failed=len(failed))
    bar.close()
    if failed:
        raise RuntimeError(f"Dukascopy years failed: {failed}. Re-run to retry only those.")
    consolidate_dukascopy()


# ----------------------------------------------------------------------
# [2/3] HistData.com M1 cross-check (2010-2023)
# ----------------------------------------------------------------------
def _histdata_year(year: int) -> pd.DataFrame:
    from histdata import download_hist_data
    from histdata.api import Platform, TimeFrame

    os.makedirs(HISTDATA_DIR, exist_ok=True)
    zips = glob.glob(os.path.join(HISTDATA_DIR, f"*M1*{year}*.zip"))
    zips = [z for z in zips if zipfile.is_zipfile(z)]
    if not zips:
        def _dl():
            p = download_hist_data(year=str(year), month=None, pair="eurusd",
                                   platform=Platform.GENERIC_ASCII,
                                   time_frame=TimeFrame.ONE_MINUTE,
                                   output_directory=HISTDATA_DIR)
            found = [p] if isinstance(p, str) and zipfile.is_zipfile(p) else \
                [z for z in glob.glob(os.path.join(HISTDATA_DIR, f"*M1*{year}*.zip")) if zipfile.is_zipfile(z)]
            if not found:
                raise RuntimeError("no valid zip produced")
            return found
        zips = _retry(_dl, f"HistData {year}")

    with zipfile.ZipFile(zips[0]) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]  # the .txt is a readme
        if len(members) != 1:
            raise RuntimeError(f"expected 1 csv in {zips[0]}, found {members}")
        with z.open(members[0]) as f:
            return _parse_histdata_csv(f)


def _parse_histdata_csv(f) -> pd.DataFrame:
    df = pd.read_csv(f, sep=";", header=None,
                     names=["timestamp", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["timestamp"], format="%Y%m%d %H%M%S")
    # HistData ASCII is fixed EST = UTC-5, no DST. Etc/GMT+5 means UTC-5 (sign is inverted).
    df["timestamp"] = ts.dt.tz_localize("Etc/GMT+5").dt.tz_convert("UTC")
    return df


def acquire_crosscheck() -> None:
    print(f"\n=== [2/3] HistData.com EURUSD 1-min cross-check ({START_YEAR}-{XVAL_LAST_YEAR}) ===")
    if _nonempty(XVAL_FILE):
        print(f" -> already exists: {XVAL_FILE}")
        return
    frames, failed = [], []
    years = list(range(START_YEAR, XVAL_LAST_YEAR + 1))
    bar = progress_bar(total=len(years), desc="HistData years", unit="year")
    for y in years:
        try:
            frames.append(_histdata_year(y))
            print(f"    [OK] {y}: {len(frames[-1]):,} rows")
        except Exception as e:  # noqa: BLE001
            print(f"    [ERROR] {y}: {e}")
            failed.append(y)
        bar.update(1)
        bar.set_postfix(year=y, failed=len(failed))
    bar.close()
    if failed:
        # Cross-check is supplementary (Appendix C); don't fabricate a partial file.
        print(f" -> [WARN] cross-check NOT written; failed years {failed}. "
              f"The primary pipeline does not depend on it.")
        return
    df = (pd.concat(frames, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last"))
    _atomic_to_csv(df, XVAL_FILE, index=False)
    print(f"    [OK] wrote {XVAL_FILE} ({len(df):,} rows, UTC)")


# ----------------------------------------------------------------------
# [3/3] FRED
# ----------------------------------------------------------------------
def _fred_one(series_id: str) -> pd.Series:
    import requests
    key = os.environ.get("FRED_API_KEY", "").strip()

    def _do():
        if key:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params={"series_id": series_id, "api_key": key, "file_type": "json",
                                     "observation_start": FRED_START, "observation_end": FRED_END},
                             timeout=(10, 60))
            r.raise_for_status()
            obs = r.json()["observations"]
            s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs}, dtype="object")
        else:
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                             params={"id": series_id, "cosd": FRED_START, "coed": FRED_END},
                             timeout=(10, 60))
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            s = pd.Series(d.iloc[:, 1].to_numpy(), index=pd.to_datetime(d.iloc[:, 0]))
        s = pd.to_numeric(s, errors="coerce")  # FRED marks missing as "."
        s.name = series_id
        s = s.dropna()
        if s.empty:
            raise ValueError("no observations")
        if s.index.max() < pd.Timestamp(EFFECTIVE_END_YEAR, 7, 1):
            raise ValueError(f"latest observation {s.index.max().date()} is too old "
                             f"for a window ending {EFFECTIVE_END_YEAR}")
        return s

    return _retry(_do, f"FRED {series_id}")


def acquire_fred(refresh: bool = False) -> None:
    print(f"\n=== [3/3] FRED ({FRED_START} -> {FRED_END}; "
          f"{'official API' if os.environ.get('FRED_API_KEY') else 'keyless CSV endpoint'}) ===")
    if _nonempty(FRED_FILE) and not refresh:
        cur = pd.read_csv(FRED_FILE, nrows=1)
        have = {c for c in cur.columns if c.lower() != "date"}
        first = pd.read_csv(FRED_FILE, usecols=[0]).iloc[:, 0].min()
        if set(FRED_SERIES) <= have and pd.Timestamp(first) <= pd.Timestamp(FRED_START) + pd.Timedelta(days=45):
            print(f" -> already exists with all series and lookback: {FRED_FILE}")
            return
        print(f" -> existing FRED file lacks series/lookback (starts {first}); re-pulling")

    series = {}
    for sid in progress_bar(FRED_SERIES, total=len(FRED_SERIES), desc="FRED series", unit="series"):
        series[sid] = _fred_one(sid)
        print(f"    [OK] {sid}: {len(series[sid]):,} obs, {series[sid].index.min().date()} "
              f"-> {series[sid].index.max().date()}")
    df = pd.concat(series.values(), axis=1).sort_index()
    df.index.name = "date"
    _atomic_to_csv(df.reset_index(), FRED_FILE, index=False)
    print(f"    [OK] wrote {FRED_FILE}")


# ----------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------
def _csv_summary(path: str):
    """Row count and first/last timestamp without loading the file."""
    with open(path, "rb") as f:
        header = f.readline()
        first = f.readline()
        n = 2
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
        f.seek(max(f.tell() - 4096, 0))
        last = f.read().strip().splitlines()[-1]
    return n - 1, first.split(b",")[0].decode(), last.split(b",")[0].decode(), header.decode().strip()


def verify_pipeline(steps) -> bool:
    print("\n=== Verification ===")
    files = []
    if "dukascopy" in steps:
        files.append(("Dukascopy master", DUKASCOPY_MASTER))
    if "fred" in steps:
        files.append(("FRED", FRED_FILE))
    ok = True
    for name, path in files:
        if not _nonempty(path):
            print(f" [FAIL] {name}: missing or empty ({path})")
            ok = False
            continue
        n, first, last, header = _csv_summary(path)
        print(f" [PASS] {name}: {n:,} rows | {first} -> {last}\n         columns: {header}")
    if "xval" in steps and not _nonempty(XVAL_FILE):
        print(f" [WARN] cross-check file not present ({XVAL_FILE}); supplementary only")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="dukascopy,xval,fred",
                    help="comma-separated subset of: dukascopy,xval,fred")
    ap.add_argument("--refresh-fred", action="store_true")
    args = ap.parse_args()
    steps = {s.strip() for s in args.steps.split(",")}

    errors = []
    for name, fn in (("dukascopy", acquire_dukascopy),
                     ("xval", acquire_crosscheck),
                     ("fred", lambda: acquire_fred(args.refresh_fred))):
        if name in steps:
            try:
                fn()
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"\n[ERROR] step '{name}' failed: {e}")
                errors.append(name)
    ok = verify_pipeline(steps)
    if errors or not ok:
        print(f"\nFAILED steps: {errors or 'verification'}")
        return 1
    print("\nAll requested ingestion steps completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
