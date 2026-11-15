"""Set which rows the poisoning positive control trains on, for the NEXT study. Never use it while a study is running.

    python -m scripts.set_positive_control --rows all            # show what would change
    python -m scripts.set_positive_control --rows all --apply    # write it, and reset the calibration

`--rows all` writes `pc_max_rows: null` (the whole training half) and `calibrated: false` in the `poisoning:` block of
config/default_config.yaml, so the next launch recalibrates the strength against the new control. Everything else in the
file, including line endings, is left as it is; a backup is written next to it.

Why: with `pc_max_rows: 100000` the control head trained on the oldest rows only and had already failed on the scored rows
before any poison was added, so check M8 failed (see src/evaluation/poisoning_protocol.pc_rows). This changes the protocol,
so it is an amendment: record it (--record-deviation) and do not apply it to an experiment that has already been analysed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = re.compile(r"(?m)^poisoning:[ \t]*\r?\n((?:[ \t]+.*\r?\n?|[ \t]*\r?\n)*)")


def rewrite(text: str, rows) -> tuple[str, list[str]]:
    """New config text and a list of the changes. rows: 'all' or a positive integer."""
    m = BLOCK.search(text)
    if not m:
        raise ValueError("no 'poisoning:' section in the config")
    value = "null" if str(rows).lower() == "all" else str(int(rows))
    blk, changes = m.group(1), []

    def setkey(block, key, new):
        pat = re.compile(rf"(?m)^([ \t]+{key}:[ \t]*)([^\s#]+)")
        mm = pat.search(block)
        if not mm:
            raise ValueError(f"poisoning.{key} is not in the config")
        if mm.group(2) != new:
            changes.append(f"poisoning.{key}: {mm.group(2)} -> {new}")
            block = block[:mm.start(2)] + new + block[mm.end(2):]
        return block
    blk = setkey(blk, "pc_max_rows", value)
    if changes:                                   # a different control invalidates the calibrated strength
        blk = setkey(blk, "calibrated", "false")
    return text[:m.start(1)] + blk + text[m.end(1):], changes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="'all' (whole training half) or a number of rows")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "default_config.yaml"))
    ap.add_argument("--apply", action="store_true", help="write the change (default: only show it)")
    a = ap.parse_args(argv)
    if str(a.rows).lower() != "all" and (not str(a.rows).isdigit() or int(a.rows) <= 0):
        print("--rows must be 'all' or a positive number")
        return 2
    with open(a.config, encoding="utf8", newline="") as fh:
        text = fh.read()
    new, changes = rewrite(text, a.rows)
    if not changes:
        print("nothing to change: the config already says that")
        return 0
    for c in changes:
        print(("changed: " if a.apply else "would change: ") + c)
    if not a.apply:
        print("add --apply to write it")
        return 0
    bak = f"{a.config}.before_positive_control_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak"
    with open(bak, "w", encoding="utf8", newline="") as fh:
        fh.write(text)
    with open(a.config, "w", encoding="utf8", newline="") as fh:
        fh.write(new)
    print(f"backup: {bak}\nnext: start the new study with --new-experiment --refreeze --record-deviation \"positive control trains on "
          f"the whole training half (M8)\"; the launcher recalibrates the strength first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
