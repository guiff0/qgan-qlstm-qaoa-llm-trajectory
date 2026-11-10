"""Fail while PREREGISTRATION_V2.md still contains unfilled decisions.   python -m scripts.check_prereg_v2"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unfilled(text: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in re.finditer(r"\[FILL:[^\]]*\]", line):
            out.append((i, m.group(0)))
    return out


def main(path: str | None = None) -> int:
    p = path or os.path.join(ROOT, "PREREGISTRATION_V2.md")
    items = unfilled(open(p, encoding="utf8").read())
    if items:
        print(f"{len(items)} decision(s) still open in {os.path.basename(p)}:")
        for ln, s in items:
            print(f"  line {ln}: {s}")
        return 1
    print("no [FILL: ...] markers left: ready to freeze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
