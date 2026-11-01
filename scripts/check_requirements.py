"""
Report which requirements are missing or at the wrong version, so the rebuild
script installs only those instead of reinstalling everything.

Usage (inside the venv):
    python scripts/check_requirements.py                 # prints unsatisfied requirement lines, one per line
    python scripts/check_requirements.py --report        # human-readable status table for every package
    python scripts/check_requirements.py --torch-cuda    # exit 0 if a CUDA build of torch is installed
    python scripts/check_requirements.py --numpy2        # exit 0 if numpy>=2 is installed

Exit code is 0 unless the flag-style checks fail; the default mode always exits 0
and simply prints what needs installing (empty output = everything satisfied).
"""
from __future__ import annotations

import argparse
import os
import sys
from importlib import metadata

try:
    from packaging.requirements import Requirement
except ImportError:  # fresh venv: packaging is vendored inside pip
    from pip._vendor.packaging.requirements import Requirement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ_FILE = os.path.join(ROOT, "requirements.txt")


def read_requirements(path: str = REQ_FILE) -> list[Requirement]:
    reqs: dict[str, Requirement] = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()      # strip inline comments
            if not line or line.startswith("-"):
                continue
            try:
                r = Requirement(line)
            except Exception:
                print(f"WARNING: cannot parse requirement line: {raw.strip()}", file=sys.stderr)
                continue
            key = r.name.lower().replace("_", "-")
            if key in reqs:                           # duplicate (e.g. requests listed twice): merge specifiers
                reqs[key].specifier &= r.specifier
            else:
                reqs[key] = r
    return list(reqs.values())


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def status(req: Requirement) -> tuple[str, str | None]:
    """Return (state, installed_version); state is ok | missing | wrong-version."""
    ver = installed_version(req.name)
    if ver is None:
        return "missing", None
    if req.specifier and not req.specifier.contains(ver, prereleases=True):
        return "wrong-version", ver
    return "ok", ver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--torch-cuda", action="store_true")
    ap.add_argument("--numpy2", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="package names to ignore")
    args = ap.parse_args()

    if args.torch_cuda:
        try:
            import torch
        except Exception:
            return 1
        return 0 if torch.version.cuda else 1

    if args.numpy2:
        v = installed_version("numpy")
        return 0 if v and int(v.split(".")[0]) >= 2 else 1

    skip = {s.lower().replace("_", "-") for s in args.skip}
    todo = []
    for req in read_requirements():
        if req.name.lower().replace("_", "-") in skip:
            continue
        state, ver = status(req)
        if args.report:
            mark = "OK  " if state == "ok" else "FIX "
            print(f"[{mark}] {req.name:<24} need '{req.specifier or 'any'}'  have {ver or '-'}")
        if state != "ok":
            todo.append(str(req))
    if not args.report:
        for t in todo:
            print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
