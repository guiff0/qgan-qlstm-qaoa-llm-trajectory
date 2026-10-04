"""
Data-sanitization gate: scans this project's own generated text
artifacts (structured .json run logs, plain .log files) for PII using
src/data/privacy_scan.py, which existed but was never called from
anywhere before this script.

SCOPE, matching privacy_scan.py's own docstring: this project's numeric
market/macro data (OHLCV, FRED series, VIX) has no PII surface at all --
there is nothing to scan there. The only place free text enters this
pipeline is logs/*.json and logs/*.log (run metadata, tracebacks,
progress messages) and, once the LLM-wiring backlog item lands,
LLM responses. This is a regex sweep for common patterns (email, IPv4,
SSN-format, phone) -- a coarse screen, not a certified PII detector; a
clean scan is evidence of absence for THESE patterns, not a guarantee.

    python -m scripts.scan_run_artifacts_for_pii
    python -m scripts.scan_run_artifacts_for_pii --log-dir logs --fail-on-hit
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.data.privacy_scan import scan_text


def _iter_log_files(log_dir: str):
    if not os.path.isdir(log_dir):
        return
    for name in sorted(os.listdir(log_dir)):
        if name.endswith(".json") or name.endswith(".log"):
            yield os.path.join(log_dir, name)


def scan_log_dir(log_dir: str) -> dict:
    """Returns {file_path: {pattern_name: count}} for every file with at
    least one hit -- a clean (empty) dict means nothing matched anywhere."""
    results = {}
    for path in _iter_log_files(log_dir):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        hits = scan_text(text)
        if any(count > 0 for count in hits.values()):
            results[path] = hits
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--fail-on-hit", action="store_true",
                     help="exit with a non-zero status if any pattern matched "
                          "(for use as a CI/pipeline gate, not just a report)")
    args = ap.parse_args()

    results = scan_log_dir(args.log_dir)
    if not results:
        print(f"[OK] No PII-pattern matches in {args.log_dir}/*.json, *.log "
              f"(email/IPv4/SSN-format/phone patterns only -- see privacy_scan.py's "
              f"docstring for what this does and doesn't catch).")
        return

    print(f"[WARN] PII-pattern matches found in {len(results)} file(s):")
    for path, hits in results.items():
        nonzero = {k: v for k, v in hits.items() if v > 0}
        print(f"  {path}: {nonzero}")

    if args.fail_on_hit:
        sys.exit(1)


if __name__ == "__main__":
    main()
