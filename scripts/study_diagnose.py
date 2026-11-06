r"""
One-sentence diagnosis for a failed study run: where it stopped, the likely cause, and the fix to try.

    d = diagnose_failure(log_path, rc, recorded=False, minutes=0.6)
    print(one_line(d))
    # why: stopped in 'attacks (FGSM/PGD/CW)': the NVIDIA key was rejected (HTTP 401) -> fix: set the key without quotes ...

It reads the tail of logs/study/<condition>_seed<k>.log (stdout+stderr of the run) and the exit code. The rules are
fixed and ordered (first match wins), so the same log always gives the same sentence. Nothing here retries or
changes anything; the study already retries failed runs on the next pass.
"""
from __future__ import annotations

import re

from scripts.study_dashboard import parse_log, read_tail

# Windows NTSTATUS exit codes seen when a native library or the console kills a process
WIN_CODES = {
    0xC000013A: ("the process was interrupted (Ctrl+C, closed console, or logoff)",
                 "just re-launch the same command; finished runs are kept and this one restarts"),
    0xC0000005: ("the process crashed natively (access violation in BLAS/torch/PennyLane, usually memory pressure "
                 "or a thread/library clash)",
                 "lower --workers (each process loads the data), close other heavy apps, and re-run; if it repeats, "
                 "rebuild the venv with rebuild_env.ps1"),
    0xC0000409: ("the process crashed natively (stack-buffer overrun in a compiled library)",
                 "lower --workers and re-run; if it repeats, rebuild the venv with rebuild_env.ps1"),
    0xC0000142: ("the process could not start (DLL initialisation failed, often too many processes for the available memory)",
                 "lower --workers and re-run"),
}

# (regex on the log tail, cause, fix). First match wins. {mod}/{err} are filled from named groups.
RULES = [
    ("key_missing", r"NvidiaApiKeyMissingError|NVIDIA_API_KEY is not set",
     "NVIDIA_API_KEY is not set in this PowerShell session",
     "run  $env:NVIDIA_API_KEY = 'nvapi-...'  (quotes are PowerShell's, not part of the key) and re-launch"),
    ("key_rejected", r"LLMAuthError|HTTP 40[13]\b.*(key|rejected)|the key was rejected",
     "the NVIDIA API rejected the key (HTTP 401/403)",
     "create a fresh key at build.nvidia.com, set it without embedded quotes, and test with scripts.check_llm_endpoint"),
    ("model_gone", r"LLMModelUnavailableError|HTTP 4(04|10) for model",
     "the configured LLM model_id is not callable on this NVIDIA account (HTTP 404/410)",
     "set llm_pipeline.model_id / fallback_model_ids in config/default_config.yaml to a model "
     "scripts.check_llm_endpoint can call, then re-launch"),
    ("api_busy", r"giving up after \d+ attempts|network error after|HTTP 429|Read timed out|ConnectionError|"
                 r"ConnectTimeout|NameResolutionError|Max retries exceeded",
     "the LLM endpoint was rate-limited or unreachable (HTTP 429/5xx, timeout, or no network)",
     "no action needed if it was temporary (the study retries on the next pass); if it repeats, lower --workers "
     "or set LLM_PROBE_RETRIES higher"),
    ("oom", r"MemoryError|Unable to allocate|not enough memory|DefaultCPUAllocator|WinError 1455|paging file is too small|"
            r"CUDA out of memory|bad allocation",
     "the process ran out of memory (RAM, page file or GPU)",
     "lower --workers (every worker loads the full data set), close other apps, or enlarge the Windows page file"),
    ("data_missing", r"Processed data not found|FileNotFoundError.*X_(train|val|test)\.npy",
     "the processed data arrays are missing",
     "run  python -m scripts.prepare_data  (RUNNING.md step 2), then re-launch"),
    ("locked", r"PermissionError|WinError 32|WinError 5\b|being used by another process|Access is denied",
     "a file (results CSV, checkpoint or log) was locked by another process",
     "close Excel/previewers on results\\ and models\\, exclude the repo from OneDrive/antivirus scanning, then "
     "re-launch (the study retries this run)"),
    ("no_module", r"(ModuleNotFoundError|ImportError): No module named '(?P<mod>[\w.]+)'",
     "the Python module '{mod}' is not importable in this environment",
     "activate the venv and run  pip install -r requirements.txt  (or rebuild_env.ps1)"),
    ("unknown_model", r"unknown model|not in models_to_run|No models selected|--only .* (matched|found) nothing|"
                      r"KeyError: '(QGAN|Classical)",
     "this condition name is not defined in the config",
     "run  python -m scripts.gen_entanglement_sweep --apply  and check the name in config/default_config.yaml "
     "(classical_ablations / ablations), then re-launch"),
    ("config", r"forecast_path|KeyError: '(?P<key>[\w.]+)'|yaml\.(scanner|parser)|ParserError|ScannerError",
     "the config is missing or has an invalid setting ({key})",
     "compare config/default_config.yaml with the last working copy (git diff) and fix the named key"),
    ("nan", r"\bNaN\b.*(loss|gradient)|loss is nan|Input contains NaN|non-finite",
     "training produced NaN/inf values",
     "re-run with the same seed to confirm; if it repeats lower the learning rate for this condition and note it as a deviation"),
]

_EXC_RE = re.compile(r"^(?:[\w.]+\.)?[A-Z]\w*(?:Error|Exception|Interrupt|Exit)\b.*$", re.M)


def _last_error_line(text: str) -> str:
    m = list(_EXC_RE.finditer(text.replace("\r", "\n")))
    return (m[-1].group(0).strip()[:160]) if m else ""


def diagnose_failure(log_path: str, rc: int, recorded: bool = False, minutes: float | None = None) -> dict:
    text = read_tail(log_path, 120_000)
    info = parse_log(log_path)
    stage = info["failed_stage"] or (info["open"][0] if info["open"] else "")
    err = _last_error_line(text)
    res = {"rule": "unknown", "stage": stage, "error_line": err}

    def done(rule, cause, fix):
        res.update(rule=rule, cause=cause, fix=fix)
        return res

    if rc == 0 and not recorded:
        return done("no_row", "the run exited cleanly but no row for this condition and seed reached results/all_results.csv",
                    "make sure results\\all_results.csv is not open or locked and the condition name matches the config, "
                    "then re-launch (the study retries)")
    code = rc & 0xFFFFFFFF if rc < 0 or rc > 255 else rc
    for rule, pat, cause, fix in RULES:
        m = re.search(pat, text, re.I | re.M)
        if m:
            gd = {k: (v or "?") for k, v in m.groupdict().items()}
            gd.setdefault("mod", "?"); gd.setdefault("key", "?")
            return done(rule, cause.format(**gd), fix.format(**gd))
    if code in WIN_CODES:
        c, f = WIN_CODES[code]
        return done(f"exit_{code:#x}", c, f)
    if not text.strip():
        return done("empty_log", f"the run died with exit {rc} before writing any output",
                    f"run  python -m src.experiments.run_all --only \"<condition>\" --seed <k>  by hand in the venv to see the error")
    if minutes is not None and minutes < 2:
        return done("startup", f"the run failed during start-up (exit {rc}, {minutes:.1f} min)"
                    + (f": {err}" if err else ""),
                    "open the log named above, read the last traceback, and fix that first; "
                    "start-up failures repeat on every retry")
    return done("other", f"unclassified failure (exit {rc})" + (f": {err}" if err else ""),
                "open the log named above and read the last traceback; re-launch to retry once")


def one_line(d: dict) -> str:
    where = f"stopped in '{d['stage']}': " if d.get("stage") else ""
    return f"why: {where}{d['cause']} -> fix: {d['fix']}"
