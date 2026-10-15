"""Wall-clock timestamps for every slow evaluation stage.

    [2026-10-12 03:41:07] START  attacks (FGSM/PGD/CW)
    [2026-10-12 03:41:48] DONE   attacks (FGSM/PGD/CW)  (41.2 s)

Printed (flushed) to stdout so a long silent stage is visible as "started at
HH:MM, still running" instead of looking frozen. Stages that raise print
FAILED with the elapsed time and re-raise unchanged."""
from __future__ import annotations

import functools
import sys
import time
from contextlib import contextmanager


def ts() -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S]")


def _fmt(sec: float) -> str:
    if sec < 120:
        return f"{sec:.1f} s"
    if sec < 7200:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.2f} h"


@contextmanager
def stage(name: str):
    t0 = time.time()
    print(f"{ts()} START  {name}", flush=True)
    try:
        yield
    except BaseException:
        print(f"{ts()} FAILED {name}  (after {_fmt(time.time() - t0)})", flush=True)
        raise
    print(f"{ts()} DONE   {name}  ({_fmt(time.time() - t0)})", flush=True)


def timed(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with stage(name):
                return fn(*a, **k)
        return wrapper
    return deco


def apply_timing(namespace: dict, names: dict) -> None:
    """Wrap, in place, every function in `names` ({attr: label}) that exists in `namespace`."""
    for attr, label in names.items():
        fn = namespace.get(attr)
        if callable(fn) and not getattr(fn, "__wrapped_by_stage__", False):
            w = timed(label)(fn)
            w.__wrapped_by_stage__ = True
            namespace[attr] = w
