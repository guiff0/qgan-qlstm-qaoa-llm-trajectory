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

import glob
import json
import os
import re
import sys
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


_EMIT_LOCK = threading.Lock()


def _emit(text: str) -> None:
    """Write a multi-line block in one call, under a lock, so blocks from different threads never interleave."""
    with _EMIT_LOCK:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


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


RUN_ID_RE = re.compile(r"run_id=([A-Za-z0-9_.\-]+)")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_STAGES = ("calibrate", "freeze", "main", "feature_map", "matrix", "hypothesis", "temporal", "screen")


def read_run_id(path: str, nbytes: int = 262144) -> str | None:
    """run_id of the LATEST attempt, from the line src.experiments.run_all writes at the start of the run
    ("Starting training run for <model> (run_id=<slug>_<UTC>)"). None until the run has reached that line."""
    try:
        with open(path, "rb") as f:
            head = f.read(nbytes).decode("utf-8", errors="replace")
    except OSError:
        return None
    m = RUN_ID_RE.findall(head.rsplit(ATTEMPT_MARK, 1)[-1])
    return m[-1] if m else None


def pipeline_lines(root: str = ROOT) -> list[str]:
    """Where the study is in its stages, and the three-arm feature-map / matrix results found on disk.
    The three-arm stages are run by scripts.run_study (stages feature_map and matrix), AFTER the main runs."""
    out = []
    try:
        with open(os.path.join(root, "results", "study_driver_state.json"), encoding="utf8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        state = {}
    marks = []
    for st in PIPELINE_STAGES:
        s = (state.get(st) or {}).get("status")
        marks.append(f"{st}={'ok' if s == 'ok' else s or 'pending'}")
    out.append("  stages: " + "  ".join(marks))
    files = sorted(glob.glob(os.path.join(root, "results", "feature_map_advantage_q*.json")))
    if not files:
        out.append("  three-arm feature-map (linear / classical RF / quantum): not run yet (stage feature_map, after the main runs)")
    for fp in files:
        label = os.path.basename(fp)[len("feature_map_advantage_"):-len(".json")]
        try:
            with open(fp, encoding="utf8") as fh:
                rows = json.load(fh).get("per_run", [])
            seeds = {r.get("seed") for r in rows}
            arms = {r.get("arm") for r in rows if r.get("arm")}
            out.append(f"  three-arm feature-map {label}: {len(seeds)} seed(s) x {len(arms) or '?'} arm(s) = {len(rows)} rows")
        except (OSError, ValueError):
            out.append(f"  three-arm feature-map {label}: file present but unreadable")
    wfiles = [f for f in sorted(glob.glob(os.path.join(root, "results", "walk_forward_q*.json"))) if "__" not in os.path.basename(f)]
    if not wfiles:
        out.append("  walk-forward windows (3 arms, same seeds): not run yet (python -m scripts.run_walk_forward)")
    for fp in wfiles:
        try:
            with open(fp, encoding="utf8") as fh:
                dd = json.load(fh)
            sm = dd.get("summary", {})
            out.append(f"  walk-forward {os.path.basename(fp)[13:-5]}: {sm.get('n_windows_run')} window(s); quantum beats both controls in "
                       f"{sm.get('windows_quantum_beats_both')}")
        except (OSError, ValueError):
            out.append(f"  walk-forward {os.path.basename(fp)}: unreadable")
    mx = os.path.join(root, "results", "three_arm_matrix.json")
    out.append("  three-arm matrix: " + ("results/three_arm_matrix.json present" if os.path.isfile(mx) else "not run yet (stage matrix)"))
    return out


class StudyDashboard:
    def __init__(self, models, seeds, remaining_fn, log_path_fn, interval: float = 120.0, experiment_id=None,
                 extra_lines_fn=None):
        self.experiment_id = experiment_id
        self.extra_lines_fn = extra_lines_fn if extra_lines_fn is not None else pipeline_lines
        self.models, self.seeds = list(models), list(seeds)
        self.remaining_fn, self.log_path_fn = remaining_fn, log_path_fn
        self.interval, self.t0 = interval, time.time()
        self.n_start = len(self.remaining_fn())
        self.active: dict[tuple, float] = {}     # (model, seed) -> start time
        self.failures: list[tuple] = []          # (model, seed, one-line diagnosis) of attempts that failed
        self._lock, self._stop = threading.Lock(), threading.Event()

    # ---- overall picture ------------------------------------------------------------------------
    def summary_lines(self, per_condition: bool = True, with_pipeline: bool = True) -> list[str]:
        """Experiment id, overall progress (done / total / remaining), per-condition counts, stage + three-arm status."""
        todo = self.remaining_fn()
        total = len(self.models) * len(self.seeds)
        done = total - len(todo)
        done_now = self.n_start - len(todo)
        elapsed = time.time() - self.t0
        eta = f"ETA {fmt_dur(elapsed / done_now * len(todo))}" if done_now > 0 and todo else "ETA n/a (no run finished yet)"
        with self._lock:
            n_running = len(self.active)
        lines = [f"[experiment] {self.experiment_id or 'legacy (no experiment id)'}",
                 f"[study] {len(self.models)} conditions x {len(self.seeds)} seeds = {total} runs; {len(todo)} remaining",
                 f"overall {bar(done, total)}  elapsed {fmt_dur(elapsed)}  {eta}",
                 f"  required {total} | done {done} | remaining {len(todo)} (running now {n_running}, queued {max(len(todo) - n_running, 0)})"]
        if per_condition:
            left = {}
            for m, _ in todo:
                left[m] = left.get(m, 0) + 1
            for m in self.models:
                lines.append(f"  {m:<34}{bar(len(self.seeds) - left.get(m, 0), len(self.seeds), 10)}")
        if with_pipeline and self.extra_lines_fn:
            try:
                lines += self.extra_lines_fn()
            except Exception:      # noqa: BLE001  -- a display bug must never stop a 12-day study
                pass
        done_set = {(m, s) for m in self.models for s in self.seeds} - set(todo)
        pending = [f for f in self.failures if (f[0], f[1]) not in done_set]
        if pending:
            lines.append(f"retry queue ({len(pending)} failed attempt(s), retried on the next pass):")
            lines += [f"  ! {m} seed {s}: {why.removeprefix('why: ')[:150]}" for m, s, why in pending[-5:]]
        return lines

    def header(self, per_condition: bool = True) -> None:
        _emit("\n" + "\n".join(self.summary_lines(per_condition)))

    # ---- per-run banners ------------------------------------------------------------------------
    def run_started(self, model: str, seed: int, log: str) -> None:
        with self._lock:
            self.active[(model, seed)] = time.time()
            idx = len(self.models) * len(self.seeds) - len(self.remaining_fn()) + len(self.active)
        _emit("\n" + "=" * 78 + f"\n>>> START  run {idx}/{len(self.models) * len(self.seeds)}  "
              f"{model}  seed {seed}\n    experiment: {self.experiment_id or 'legacy'}   "
              f"run_id: assigned when the run starts (shown in the next status block)\n    log: {log}\n" + "=" * 78)

    def run_finished(self, model: str, seed: int, ok: bool, rc: int, log: str, diagnosis: str = "") -> None:
        with self._lock:
            t = self.active.pop((model, seed), time.time())
        info = parse_log(log)
        verdict = "OK  " if ok else "FAIL"
        lines = [f"<<< {verdict}  {model}  seed {seed}  exit {rc}  in {fmt_dur(time.time() - t)}  "
                 f"({info['n_done']} subtasks done)",
                 f"    experiment: {self.experiment_id or 'legacy'}   run_id: {read_run_id(log) or 'n/a'}"]
        if not ok:
            if diagnosis:
                lines.append(f"    {diagnosis}")
                self.failures.append((model, seed, diagnosis))
            lines.append(f"    log: {log}")
        _emit("\n" + "-" * 78 + "\n" + "\n".join(lines) + "\n" + "-" * 78)
        self.header()

    # ---- live subtasks --------------------------------------------------------------------------
    def snapshot(self) -> None:
        with self._lock:
            act = dict(self.active)
        if not act:
            return
        # ONE write for the whole block: several threads and the run logs share one console, and separate print()
        # calls interleave there (that is what garbled the old block).
        out = ["", "~" * 30 + f" status {time.strftime('%H:%M:%S')} " + "~" * 30] + self.summary_lines(per_condition=True)
        out.append("running now:")
        for (model, seed), t in sorted(act.items(), key=lambda kv: kv[1]):
            log = self.log_path_fn(model, seed)
            info = parse_log(log)
            out.append(f"  {model} seed {seed}   running {fmt_dur(time.time() - t)}   run_id: {read_run_id(log) or 'starting...'}")
            if info["open"]:
                out.append(f"      subtask {info['n_done'] + 1}: {info['open'][0]}   since {info['open'][1][11:]}   "
                           f"| {info['n_done']} subtasks done")
            else:
                out.append(f"      between subtasks   | {info['n_done']} subtasks done")
            if info["progress"]:
                out.append(f"      last progress: {info['progress']}")
        _emit("\n".join(out) + "\n")

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
