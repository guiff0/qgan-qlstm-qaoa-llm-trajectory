"""Pilot table: DESCRIPTIVE ONLY. One seed per model, no test, no interval, no verdict.

    python -m scripts.pilot_table                      # prints the table, writes results/pilot_table.md and .json
    python -m scripts.pilot_table --all-runs           # also list the earlier (superseded) run of each model

What it is for: the models that were run once in October (Classical LSTM, QLSTM Forecaster, QLSTM-LLM, QAOA-Enhanced LLM,
QAOA-LLM, Classical-Features-LLM, LLM Forecaster) are not conditions of the pre-registered study and no hypothesis names
them, so they cannot appear in the verdict table. They are reported here as RMSE divided by the persistence ("no change")
RMSE on the same test rows: 1.0 = no better than predicting no change, below 1 = better, above 1 = worse.

What it is NOT: evidence for or against any hypothesis. With one seed there is no variance estimate, so "A is better than B"
cannot be read from this table. scripts.run_hypothesis_tests never reads these rows and this script never reads or writes a
hypothesis, verdict, walk-forward, naive-baseline or feature-map file. It imports nothing from the three-arm study.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import pandas as pd

# slug -> (display name, what the model is, what its forecast head regresses)
PILOT = {
    "classical_lstm": ("Classical LSTM", "LSTM forecaster; no generator, no LLM, no quantum part", "price level"),
    "qlstm_forecaster": ("QLSTM Forecaster", "quantum-circuit forecaster trained on the target; no generator, no LLM", "next-bar change"),
    "qlstm_llm": ("QLSTM-LLM", "QLSTM forecast placed in the LLM prompt", "next-bar change"),
    "qaoa_enhanced_llm": ("QAOA-Enhanced LLM", "one-time QAOA feature selection, then a linear head; NO LLM call despite the name",
                          "price level"),
    "qaoa_llm": ("QAOA-LLM", "QAOA-selected features named in the LLM prompt", "LLM forecast"),
    "classical_features_llm": ("Classical-Features-LLM", "classically selected features named in the LLM prompt (control for QAOA-LLM)",
                               "LLM forecast"),
    "llm_forecaster": ("LLM Forecaster", "LLM alone, no generator, no feature selection", "LLM forecast"),
}
STAMP = re.compile(r"_(\d{8}T\d{6}Z)$")
BANNER = "PILOT, DESCRIPTIVE ONLY: one seed per model, no inference, not part of any hypothesis verdict."


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def build(df: pd.DataFrame) -> dict:
    """Latest run per pilot model plus its earlier runs. Rows of the pre-registered conditions are never touched."""
    rows = []
    if "run_id" not in df.columns:
        return {"banner": BANNER, "rows": rows}
    d = df.copy()
    d["_slug"] = d["run_id"].astype(str).map(lambda s: STAMP.sub("", s))
    d["_utc"] = d["run_id"].astype(str).map(lambda s: (STAMP.search(s) or [None, ""])[1])
    for slug, (name, what, frame) in PILOT.items():
        g = d[d["_slug"] == slug].sort_values("_utc")
        if g.empty:
            continue
        seeds = sorted({int(s) for s in g["seed"].dropna()}) if "seed" in g.columns else []

        def one(r):
            rmse, pers = _num(r.get("rmse")), _num(r.get("persistence_rmse"))
            ratio = _num(r.get("rmse_over_persistence")) or (rmse / pers if rmse is not None and pers else None)
            sel = r.get("selected_features")
            sel = [s for s in str(sel).split(",") if s] if isinstance(sel, str) else None
            tf = r.get("target_frame")       # recorded by the row itself when present; else the model's documented frame
            regresses = {"delta": "next-bar change", "level": "price level"}.get(tf if isinstance(tf, str) else None, frame)
            return {"run_id": r["run_id"], "run_utc": r["_utc"], "seed": None if pd.isna(r.get("seed")) else int(r["seed"]),
                    "head_regresses": regresses,
                    "rmse": rmse, "persistence_rmse": pers, "rmse_over_persistence": ratio,
                    "beats_persistence": None if ratio is None else bool(ratio < 1.0),
                    "llm_rmse_over_persistence": _num(r.get("llm_rmse_over_persistence")),
                    "llm_n_forecasts": None if _num(r.get("llm_n_forecasts")) is None else int(_num(r.get("llm_n_forecasts"))),
                    "n_selected_features": len(sel) if sel else None, "selected_features": sel}
        runs = [one(r) for _, r in g.iterrows()]
        rows.append({"slug": slug, "model": name, "what": what, "head_regresses": frame, "n_runs": len(runs),
                     "n_distinct_seeds": len(seeds), "seeds": seeds, "latest": runs[-1], "earlier": runs[:-1]})
    return {"banner": BANNER, "reference": "persistence (no change) = 1.000 by definition", "rows": rows,
            "reading": ["ratio = model RMSE / persistence RMSE on the same test rows; below 1 beats 'no change'",
                        "n = 1 seed per model: no variance, so no model-to-model comparison can be drawn",
                        "a row whose head regresses the price LEVEL (Classical LSTM and QAOA-Enhanced LLM before 14 Nov 2026) "
                        "carries the train-to-test price shift, which is why its ratio is tens to hundreds: that is a property "
                        "of the frame, not a finding about the model. Rows marked 'next-bar change' are like-for-like",
                        "the LLM column is scored on about 100 test windows, not the full test set"]}


def _r(v, nd=3):
    return "n/a" if v is None else (f"{v:,.{nd}f}" if abs(v) < 1e6 else f"{v:.3g}")


def to_markdown(tbl: dict, all_runs: bool = False) -> str:
    L = [f"**{tbl['banner']}**", "", "| Model | What it is | Head regresses | Seed | Run (UTC) | RMSE / persistence | Beats no-change | LLM RMSE / persistence (windows) |",
         "|---|---|---|---|---|---|---|---|"]
    for row in tbl["rows"]:
        for i, r in enumerate([row["latest"]] + (row["earlier"][::-1] if all_runs else [])):
            llm = "n/a" if r["llm_rmse_over_persistence"] is None else f"{_r(r['llm_rmse_over_persistence'])} ({r['llm_n_forecasts']})"
            L.append(f"| {row['model'] if i == 0 else row['model'] + ' (earlier, superseded)'} | {row['what'] if i == 0 else ''} | "
                     f"{r['head_regresses']} | {r['seed']} | {r['run_utc']} | {_r(r['rmse_over_persistence'])} | "
                     f"{'n/a' if r['beats_persistence'] is None else ('yes' if r['beats_persistence'] else 'no')} | {llm} |")
    L += ["", "Persistence (no change) = 1.000 by definition.", ""] + [f"- {x}" for x in tbl["reading"]]
    sel = [f"- {row['model']}: {row['latest']['n_selected_features']} feature(s) selected ({', '.join(row['latest']['selected_features'])})"
           for row in tbl["rows"] if row["latest"]["selected_features"]]
    if sel:
        L += ["", "Features the selector actually returned:"] + sel
    multi = [row["model"] for row in tbl["rows"] if row["n_distinct_seeds"] > 1]
    if multi:
        L += ["", f"NOTE: more than one seed found for {', '.join(multi)}; this table still shows the latest run only."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-csv", default=os.path.join("results", "all_results.csv"))
    ap.add_argument("--out", default=os.path.join("results", "pilot_table"), help="path stem; .md and .json are written")
    ap.add_argument("--all-runs", action="store_true", help="also list each model's earlier (superseded) runs")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.results_csv):
        print(f"no results file at {a.results_csv}")
        return 2
    tbl = build(pd.read_csv(a.results_csv, low_memory=False))
    if not tbl["rows"]:
        print("no pilot rows found (none of: " + ", ".join(PILOT) + ")")
        return 1
    md = to_markdown(tbl, a.all_runs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".md", "w", encoding="utf8") as fh:
        fh.write(md)
    with open(a.out + ".json", "w", encoding="utf8") as fh:
        json.dump(tbl, fh, indent=2)
    print(md)
    print(f"written: {a.out}.md, {a.out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
