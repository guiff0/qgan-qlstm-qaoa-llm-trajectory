r"""
Console dashboard for scripts/run_full_study.py (append-only output, safe in PowerShell and when redirected).

What it prints
  * a header every time the picture changes:
        [study] 8 conditions x 10 seeds = 80 runs; 80 remaining
        overall [######----------------------]  20/80  25%  elapsed 6.2 h  ETA 18.7 h
        QGAN-LLM                      ###.......  3/10
  * a banner when each run starts and when it ends (OK / FAIL, minutes, what it was doing when it failed)
  * every `interval` seconds, one block listing each ACTIVE run with its subtask, taken from the per-run
    log that src/utils/stage_log.py already writes ("[time] START <stage>" / "DONE <stage> (41.2 s)"):
        run 3/80  QGAN-LLM (ent-L6 full-d4) seed 46   running 3.1 h
            subtask 7: attacks (FGSM/PGD/CW)  since 03:41:07 (12.3 min)   | 6 subtasks done
            last progress: Epoch 14/50 ... 28%
"""
from __future__ import annotations

import os
import re
import threading
import time

STAGE_RE = re.compile(r"\[(?P<t>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]\s+(?P<ev>START|DONE|FAILED)\s+(?P<name>.*?)"
                      r"(?:\s+\((?:after )?(?P<dur>[\d.]+ (?:s|min|h))\))?\s*$")
BAR_RE = re.compile(r"\d+%\|")


def bar(done: int, total: int, width: int = 28, fill: str = "#", empty: str = "-") -> str:
    total = max(total, 1)
    n = int(round(width * min(done, total) / total))
    return f"[{fill * n}{empty * (width - n)}] {done:>3}/{total}  {100 * done // total:>3}%"


def fmt_dur(sec: float) -> str:
    if sec < 120:
        return f"{sec:.0f} s"
    if sec < 7200:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.1f} h"


ATTEMPT_MARK = "===== attempt started"


def read_tail(path: str, nbytes: int = 65536) -> str:
    """Last `nbytes` of a run log, cut to the LATEST attempt (a retried run appends to the same log)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - nbytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return text.rsplit(ATTEMPT_MARK, 1)[-1]


def parse_log(path: str) -> dict:
    """Subtask state from a run log: finished stages, the stage currently open, the last progress-bar fragment."""
    text = read_tail(path)
    open_stage, n_done, failed = None, 0, None
    for line in text.replace("\r", "\n").splitlines():
        m = STAGE_RE.search(line)
        if not m:
            continue
        ev, name = m.group("ev"), m.group("name").strip()
        if ev == "START":
            open_stage = (name, m.group("t"))
        elif ev == "DONE":
            n_done += 1
            if open_stage and open_stage[0] == name:
                open_stage = None
        else:
            failed, open_stage = name, None
    bars = [c for c in text.replace("\n", "\r").split("\r") if BAR_RE.search(c)]
    return {"open": open_stage, "n_done": n_done, "failed_stage": failed,
            "progress": bars[-1].strip()[:110] if bars else ""}


class StudyDashboard:
    def __init__(self, models, seeds, remaining_fn, log_path_fn, interval: float = 120.0, experiment_id=None):
        self.experiment_id = experiment_id
        self.models, self.seeds = list(models), list(seeds)
        self.remaining_fn, self.log_path_fn = remaining_fn, log_path_fn
        self.interval, self.t0 = interval, time.time()
        self.n_start = len(self.remaining_fn())
        self.active: dict[tuple, float] = {}     # (model, seed) -> start time
        self.failures: list[tuple] = []          # (model, seed, one-line diagnosis) of attempts that failed
        self._lock, self._stop = threading.Lock(), threading.Event()

    # ---- overall picture ------------------------------------------------------------------------
    def header(self, per_condition: bool = True) -> None:
        todo = self.remaining_fn()
        total = len(self.models) * len(self.seeds)
        done = total - len(todo)
        done_now = self.n_start - len(todo)
        elapsed = time.time() - self.t0
        eta = f"ETA {fmt_dur(elapsed / done_now * len(todo))}" if done_now > 0 and todo else "ETA n/a (no run finished yet)"
        lines = ([f"[experiment] {self.experiment_id}"] if self.experiment_id else []) + [
                 f"[study] {len(self.models)} conditions x {len(self.seeds)} seeds = {total} runs; {len(todo)} remaining",
                 f"overall {bar(done, total)}  elapsed {fmt_dur(elapsed)}  {eta}"]
        if per_condition:
            left = {}
            for m, _ in todo:
                left[m] = left.get(m, 0) + 1
            for m in self.models:
                lines.append(f"  {m:<34}{bar(len(self.seeds) - left.get(m, 0), len(self.seeds), 10)}")
        done_set = {(m, s) for m in self.models for s in self.seeds} - set(todo)
        pending = [f for f in self.failures if (f[0], f[1]) not in done_set]
        if pending:
            lines.append(f"retry queue ({len(pending)} failed attempt(s), retried on the next pass):")
            lines += [f"  ! {m} seed {s}: {why.removeprefix('why: ')[:150]}" for m, s, why in pending[-5:]]
        print("\n" + "\n".join(lines), flush=True)

    # ---- per-run banners ------------------------------------------------------------------------
    def run_started(self, model: str, seed: int, log: str) -> None:
        with self._lock:
            self.active[(model, seed)] = time.time()
            idx = len(self.models) * len(self.seeds) - len(self.remaining_fn()) + len(self.active)
        print("\n" + "=" * 78 + f"\n>>> START  run {idx}/{len(self.models) * len(self.seeds)}  "
              f"{model}  seed {seed}\n    log: {log}\n" + "=" * 78, flush=True)

    def run_finished(self, model: str, seed: int, ok: bool, rc: int, log: str, diagnosis: str = "") -> None:
        with self._lock:
            t = self.active.pop((model, seed), time.time())
        info = parse_log(log)
        verdict = "OK  " if ok else "FAIL"
        lines = [f"<<< {verdict}  {model}  seed {seed}  exit {rc}  in {fmt_dur(time.time() - t)}  "
                 f"({info['n_done']} subtasks done)"]
        if not ok:
            if diagnosis:
                lines.append(f"    {diagnosis}")
                self.failures.append((model, seed, diagnosis))
            lines.append(f"    log: {log}")
        print("\n" + "-" * 78 + "\n" + "\n".join(lines) + "\n" + "-" * 78, flush=True)
        self.header()

    # ---- live subtasks --------------------------------------------------------------------------
    def snapshot(self) -> None:
        with self._lock:
            act = dict(self.active)
        if not act:
            return
        print("\n" + "~" * 30 + f" status {time.strftime('%H:%M:%S')} " + "~" * 30, flush=True)
        for (model, seed), t in sorted(act.items(), key=lambda kv: kv[1]):
            info = parse_log(self.log_path_fn(model, seed))
            print(f"  {model} seed {seed}   running {fmt_dur(time.time() - t)}")
            if info["open"]:
                print(f"      subtask {info['n_done'] + 1}: {info['open'][0]}   since {info['open'][1][11:]}   "
                      f"| {info['n_done']} subtasks done")
            else:
                print(f"      between subtasks   | {info['n_done']} subtasks done")
            if info["progress"]:
                print(f"      last progress: {info['progress']}")
        print(flush=True)

    def start_monitor(self) -> None:
        def loop():
            while not self._stop.wait(self.interval):
                try:
                    self.snapshot()
                except Exception:      # noqa: BLE001  -- a display bug must never stop a 12-day study
                    pass
        threading.Thread(target=loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
