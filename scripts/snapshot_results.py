"""Snapshot results/all_results.csv and record its sha256, in one command (gate step 2 of the confirmatory protocol).

  python -m scripts.snapshot_results                 # copy -> results/snapshots/all_results_<UTC>.csv + sha256 + manifest
  python -m scripts.snapshot_results --verify        # exit 0 if the CURRENT csv is byte-identical to a recorded snapshot
  python -m scripts.snapshot_results --note "last run finished"   # free-text note stored in the manifest

The snapshot is made read-only and never overwritten; results/snapshots/manifest.json is append-only. Taking a snapshot of a
csv that is already snapshotted (same sha256) reuses the existing one. run_hypothesis_tests STOPS when the study is
complete, takes this snapshot by itself, and only runs the confirmatory tests (--run-confirmatory) if the csv still
matches a snapshot: edit the data afterwards and the gate refuses until you snapshot again and disclose it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join("results", "all_results.csv")
SNAP_DIR = os.path.join(ROOT, "results", "snapshots")
MANIFEST = os.path.join(SNAP_DIR, "manifest.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _abs(csv: str) -> str:
    return csv if os.path.isabs(csv) else os.path.join(ROOT, csv)


def load_manifest() -> list:
    try:
        with open(MANIFEST, encoding="utf8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


def _save_manifest(entries: list) -> None:
    os.makedirs(SNAP_DIR, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf8") as fh:
        json.dump(entries, fh, indent=2)
    os.replace(tmp, MANIFEST)


def snapshot(csv: str = DEFAULT_CSV, note: str = "") -> dict:
    src = _abs(csv)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{src} does not exist: nothing to snapshot")
    digest = sha256_file(src)
    entries = load_manifest()
    for e in entries:
        if e["sha256"] == digest and os.path.isfile(os.path.join(ROOT, e["snapshot"])):
            return dict(e, reused=True)
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(SNAP_DIR, exist_ok=True)
    dest = os.path.join(SNAP_DIR, f"all_results_{utc}.csv")
    with open(src, "rb") as a, open(dest, "xb") as b:          # "x": never overwrite a snapshot
        shutil.copyfileobj(a, b)
    if sha256_file(dest) != digest:
        os.remove(dest)
        raise RuntimeError("snapshot copy does not match the source (the csv changed while copying?); try again")
    with open(dest + ".sha256", "w", encoding="utf8") as fh:
        fh.write(f"{digest}  {os.path.basename(dest)}\n")
    for p in (dest, dest + ".sha256"):
        os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    with open(dest, "rb") as fh:
        n_rows = max(sum(1 for _ in fh) - 1, 0)
    entry = {"utc": utc, "snapshot": os.path.relpath(dest, ROOT).replace(os.sep, "/"), "source": csv.replace(os.sep, "/"),
             "sha256": digest, "bytes": os.path.getsize(dest), "rows": n_rows, "note": note}
    _save_manifest(entries + [entry])
    return dict(entry, reused=False)


def verify(csv: str = DEFAULT_CSV) -> tuple[bool, dict | None]:
    """(True, entry) when the current csv is byte-identical to a recorded snapshot."""
    src = _abs(csv)
    if not os.path.isfile(src):
        return False, None
    digest = sha256_file(src)
    for e in reversed(load_manifest()):
        if e["sha256"] == digest:
            return True, e
    return False, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--note", default="")
    ap.add_argument("--verify", action="store_true", help="only check that the current csv matches a snapshot")
    a = ap.parse_args(argv)
    if a.verify:
        ok, e = verify(a.csv)
        print(f"csv matches snapshot {e['snapshot']} (sha256 {e['sha256']})" if ok else
              "the current csv does NOT match any recorded snapshot (it changed, or none was taken): "
              "python -m scripts.snapshot_results")
        return 0 if ok else 1
    try:
        e = snapshot(a.csv, a.note)
    except (OSError, RuntimeError) as err:
        print(f"snapshot failed: {err}")
        return 2
    print(("already snapshotted (same sha256): " if e["reused"] else "snapshot written: ") + e["snapshot"])
    print(f"sha256  {e['sha256']}\nrows    {e['rows']}   bytes {e['bytes']}\nmanifest {os.path.relpath(MANIFEST, ROOT)}")
    print("Do not edit the csv or the snapshot from here on; the confirmatory run checks that they still match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
