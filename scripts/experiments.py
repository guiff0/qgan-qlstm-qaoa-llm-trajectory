r"""Manage experiments: the id that groups many run_ids so two studies are never confused.

    python -m scripts.experiments list                       # every experiment known here + rows per id in the CSV
    python -m scripts.experiments show EXP-20261107-WINBOX-3fa9
    python -m scripts.experiments verify EXP-20261107-WINBOX-3fa9      # plan vs rows: missing / duplicate / outside plan
    python -m scripts.experiments adopt --since 2026-11-03 --apply     # stamp earlier UNLABELLED rows with a new id
    python -m scripts.experiments import D:\from_machine_b.zip --apply # bring another machine's experiment here

Start a study:   .\scripts\run_full_study.ps1 --new-experiment --label "machine A"
Second machine:  .\scripts\run_full_study.ps1 --new-experiment --label "machine B" --base-seed 143   (distinct seeds)

adopt / import never delete anything: all_results.csv is archived (results/history/) before it is rewritten, the
append-only ledger is untouched, and both commands print what they would do and change nothing without --apply.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import experiment as exp  # noqa: E402

CSV = os.path.join("results", "all_results.csv")


# ------------------------------------------------------------------ csv ----
def read_csv(path: str) -> tuple[list[str], list[dict]]:
    if not os.path.isfile(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    from src.utils.io import atomic_write_text
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    atomic_write_text(path, buf.getvalue())


def _slug(row: dict) -> str:
    return re.sub(r"_\d{8}T\d{6}Z$", "", row.get("run_id", ""))


def _seed(row: dict):
    try:
        return int(float(row.get("seed", "")))
    except ValueError:
        return None


# ---------------------------------------------------------------- adopt ----
def adopt(eid: str, since: str, *, conditions, seeds, base_seed: int, label: str = "", only_seeds=None,
          csv_path: str = CSV, root: str = exp.EXPERIMENT_ROOT, apply: bool = False, config_path=None) -> dict:
    """Give earlier UNLABELLED rows (run_utc >= since, matching the plan's conditions and seeds) the experiment id."""
    fields, rows = read_csv(csv_path)
    want_slugs = {exp._slug(c) for c in conditions}
    want_seeds = set(only_seeds if only_seeds is not None else seeds)
    pick = [r for r in rows if not (r.get("experiment_id") or "").strip()
            and (r.get("run_utc") or "") >= since and _slug(r) in want_slugs and _seed(r) in want_seeds]
    per_cond: dict[str, int] = {}
    for r in pick:
        per_cond[_slug(r)] = per_cond.get(_slug(r), 0) + 1
    report = {"experiment_id": eid, "rows_to_stamp": len(pick), "per_condition": per_cond,
              "unlabelled_rows_left_alone": sum(1 for r in rows if not (r.get("experiment_id") or "").strip()) - len(pick),
              "applied": False}
    if not apply or not pick:
        return report
    from src.utils.run_identity import archive_file
    report["csv_archive"] = archive_file(csv_path, tag=f"before_adopt_{eid}")
    exp.create_experiment(eid, conditions=conditions, seeds=seeds, base_seed=base_seed, label=label,
                          config_path=config_path, root=root)
    if "experiment_id" not in fields:
        fields.append("experiment_id")
    ids = {id(r) for r in pick}
    for r in rows:
        if id(r) in ids:
            r["experiment_id"] = eid
    write_csv(csv_path, fields, rows)
    for r in pick:
        exp.register_run(eid, {"condition": r.get("model_type"), "slug": _slug(r), "seed": _seed(r), "status": "ok",
                               "run_id": r.get("run_id"), "adopted": True, "run_utc": r.get("run_utc")}, root)
    exp.set_active(eid, root)
    report["applied"] = True
    return report


# --------------------------------------------------------------- import ----
class ImportConflict(RuntimeError):
    pass


def _find_results_dir(base: str) -> str:
    for cand in (base, os.path.join(base, "results")):
        if os.path.isfile(os.path.join(cand, "all_results.csv")):
            return cand
    raise ImportConflict(f"no all_results.csv found in {base} (or {base}/results)")


def import_experiment(src: str, *, only_id: str | None = None, csv_path: str = CSV,
                      root: str = exp.EXPERIMENT_ROOT, apply: bool = False) -> dict:
    """Merge another machine's experiment(s) into this results folder. Idempotent; refuses real conflicts."""
    tmp = None
    if zipfile.is_zipfile(src):
        tmp = tempfile.mkdtemp(prefix="exp_import_")
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    try:
        rdir = _find_results_dir(src)
        sfields, srows = read_csv(os.path.join(rdir, "all_results.csv"))
        ids = sorted({(r.get("experiment_id") or "").strip() for r in srows} - {""})
        if only_id:
            ids = [i for i in ids if i == only_id]
        if not ids:
            raise ImportConflict("the source has no rows with an experiment_id (run `adopt` on that machine first); "
                                 "unlabelled rows are never imported because they cannot be told apart from local ones")
        fields, rows = read_csv(csv_path)
        local = {r.get("run_id"): r for r in rows}
        report = {"experiments": {}, "applied": False}
        add: list[dict] = []
        for eid in ids:
            src_rows = [r for r in srows if (r.get("experiment_id") or "").strip() == eid]
            new, same = [], 0
            for r in src_rows:
                old = local.get(r.get("run_id"))
                if old is None:
                    new.append(r)
                elif (old.get("experiment_id") or "").strip() != eid:
                    raise ImportConflict(f"run_id {r.get('run_id')} exists locally under experiment "
                                         f"{old.get('experiment_id') or 'unassigned'}, not {eid}")
                elif all((old.get(k) or "") == (r.get(k) or "") for k in sfields if k in old or r.get(k)):
                    same += 1
                else:
                    raise ImportConflict(f"run_id {r.get('run_id')} exists locally with different values")
            sman = os.path.join(rdir, "experiments", eid, "manifest.json")
            lman = os.path.join(root, eid, "manifest.json")
            if os.path.isfile(sman) and os.path.isfile(lman):
                a, b = json.load(open(sman)), json.load(open(lman))
                if a.get("plan_hash") != b.get("plan_hash"):
                    raise ImportConflict(f"{eid}: manifest plan differs between the two machines")
            report["experiments"][eid] = {"rows_in_source": len(src_rows), "new_rows": len(new), "already_here": same,
                                          "manifest_in_source": os.path.isfile(sman)}
            add += new
        if not apply:
            return report
        from src.utils.run_identity import archive_file, append_ledger
        report["csv_archive"] = archive_file(csv_path, tag="before_import")
        for k in sfields:
            if k not in fields:
                fields.append(k)
        write_csv(csv_path, fields, rows + add)
        for r in add:
            append_ledger(os.path.dirname(csv_path) or "results", {k: v for k, v in r.items() if v != ""})
        for eid in ids:
            sdir = os.path.join(rdir, "experiments", eid)
            if os.path.isdir(sdir):
                dst = exp.experiment_dir(eid, root)
                os.makedirs(dst, exist_ok=True)
                for name in ("manifest.json",):
                    if os.path.isfile(os.path.join(sdir, name)) and not os.path.isfile(os.path.join(dst, name)):
                        shutil.copy2(os.path.join(sdir, name), os.path.join(dst, name))
                have = {json.dumps(x, sort_keys=True) for x in exp.read_runs(eid, root)}
                for x in _read_jsonl(os.path.join(sdir, "runs.jsonl")):
                    if json.dumps(x, sort_keys=True) not in have:
                        with open(os.path.join(dst, "runs.jsonl"), "a", encoding="utf-8") as f:
                            f.write(json.dumps(x) + "\n")
            pdir = os.path.join(rdir, "predictions", eid)
            if os.path.isdir(pdir):
                shutil.copytree(pdir, os.path.join(os.path.dirname(csv_path) or "results", "predictions", eid),
                                dirs_exist_ok=True)
        report["applied"] = True
        return report
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# --------------------------------------------------------------- verify ----
def verify(eid: str, csv_path: str = CSV, root: str = exp.EXPERIMENT_ROOT) -> dict:
    man = exp.load_manifest(eid, root)
    _, rows = read_csv(csv_path)
    mine = [r for r in rows if ((r.get("experiment_id") or "").strip() or "unassigned") == eid]
    plan = {(exp._slug(c), s) for c in man["conditions"] for s in man["seeds"]}
    count: dict[tuple, int] = {}
    for r in mine:
        k = (_slug(r), _seed(r))
        count[k] = count.get(k, 0) + 1
    runs = exp.read_runs(eid, root)
    ok_ids = {x.get("run_id") for x in runs if x.get("status") == "ok" and x.get("run_id")}
    csv_ids = {r.get("run_id") for r in mine}
    rep = {"experiment_id": eid, "planned": len(plan), "done": len(plan & set(count)),
           "missing": sorted(f"{c} seed {s}" for c, s in plan - set(count)),
           "duplicates": sorted(f"{c} seed {s} x{n}" for (c, s), n in count.items() if n > 1 and (c, s) in plan),
           "outside_plan": sorted(f"{c} seed {s}" for (c, s) in count if (c, s) not in plan),
           "logged_ok_without_csv_row": sorted(ok_ids - csv_ids),
           "csv_rows_not_in_runs_log": sorted(csv_ids - ok_ids)[:20],
           "code_changed_since_start": sorted(p for p, h in (man.get("code_sha256") or {}).items()
                                              if h and exp._sha(p) not in (None, h))}
    rep["clean"] = not (rep["outside_plan"] or rep["logged_ok_without_csv_row"])
    return rep


# ------------------------------------------------------------------ cli ----
def _plan_args(ap) -> None:
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=43)
    ap.add_argument("--levels", default="0,2,4,6")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("show"); p.add_argument("id")
    p = sub.add_parser("verify"); p.add_argument("id")
    p = sub.add_parser("adopt", help="stamp earlier unlabelled rows with an experiment id")
    p.add_argument("--id", default=None, help="default: a new id")
    p.add_argument("--since", required=True, help="only rows with run_utc >= this (e.g. 2026-11-03)")
    p.add_argument("--label", default="adopted from the first launch")
    p.add_argument("--config", default="config/default_config.yaml")
    _plan_args(p); p.add_argument("--apply", action="store_true")
    p = sub.add_parser("import", help="merge another machine's experiment (folder or zip)")
    p.add_argument("path"); p.add_argument("--id", default=None); p.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        import pandas as pd
        counts = exp.experiments_in(pd.read_csv(CSV)) if os.path.isfile(CSV) else {}
        mans = {m["experiment_id"]: m for m in exp.list_experiments()}
        active = exp.get_active()
        print(f"{'experiment':<30}{'rows':>6}{'planned':>9}  {'machine':<14}{'created':<18}label")
        for eid in sorted(set(counts) | set(mans)):
            m = mans.get(eid, {})
            print(f"{eid + (' *' if eid == active else ''):<30}{counts.get(eid, 0):>6}{m.get('planned_runs', '-'):>9}  "
                  f"{(m.get('machine') or {}).get('name', '-'):<14}{m.get('created_utc', '-'):<18}{m.get('label', '')}")
        print("(* = active: the one a re-launch without flags resumes)")
        return 0
    if a.cmd == "show":
        man = exp.load_manifest(a.id)
        print(json.dumps({k: v for k, v in man.items() if k != "code_sha256"}, indent=2))
        runs = exp.read_runs(a.id)
        print(f"\n{len(runs)} logged attempts; run ids of successful runs:")
        for r in runs:
            if r.get("status") == "ok":
                print(f"  {r.get('slug'):<34} seed {r.get('seed'):<4} {r.get('run_id')}")
        return 0
    if a.cmd == "verify":
        rep = verify(a.id)
        print(json.dumps(rep, indent=2))
        return 0 if rep["clean"] else 1
    if a.cmd == "adopt":
        from scripts.run_full_study import build_conditions, parse_levels
        models = build_conditions(parse_levels(a.levels))
        seeds = [a.base_seed + i for i in range(a.n_runs)]
        eid = a.id or exp.new_experiment_id()
        rep = adopt(eid, a.since, conditions=models, seeds=seeds, base_seed=a.base_seed, label=a.label,
                    apply=a.apply, config_path=a.config)
        print(json.dumps(rep, indent=2))
        if not a.apply:
            print("\nDry run: nothing changed. Add --apply to stamp these rows.")
        return 0
    if a.cmd == "import":
        try:
            rep = import_experiment(a.path, only_id=a.id, apply=a.apply)
        except ImportConflict as exc:
            print(f"REFUSED: {exc}")
            return 2
        print(json.dumps(rep, indent=2))
        if not a.apply:
            print("\nDry run: nothing changed. Add --apply to merge.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
