"""Run the SEC technical-control screen over run JSON files.

usage: python scripts/sec_compliance_check.py run1.json run2.json ... [--config config/default_config.yaml]
                                      [--attest attestations.json] [--paper paper.txt] [--out results]
Writes results/sec_compliance_report.{json,md}. Exit 1 on any FAIL only with --strict.
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.sec_compliance import evaluate_run, cross_run_checks  # noqa: E402

# Unsupported-claim patterns for the manuscript text.
CLAIM_PATTERNS = [
    (r"(?i)15c3-5[^.]{0,200}?(\d+(?:\.\d+)?)\s*(?:ms|millisecond)", "15c3-5 has no numeric latency limit"),
    (r"(?i)(?:mandat\w+|required?)[^.]{0,80}(?:40\s*ms|sub-?millisecond)", "latency 'mandate' is not in any SEC rule"),
    (r"(?i)reg(?:ulation)?\s*sci[^.]{0,120}(?:this|our|the) (?:model|framework|study)", "Reg SCI applies to SCI entities only"),
]


def scan_paper(text):
    hits = []
    for pat, why in CLAIM_PATTERNS:
        for mt in re.finditer(pat, text):
            hits.append({"why": why, "text": text[max(0, mt.start()-40): mt.end()+40].replace("\n", " ")})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--config"); ap.add_argument("--attest"); ap.add_argument("--paper")
    ap.add_argument("--out", default="results"); ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    cfg = {}
    if a.config:
        import yaml
        cfg = (yaml.safe_load(open(a.config)) or {}).get("sec_compliance", {})
    att = json.load(open(a.attest)) if a.attest else {}
    reports = []
    for f in a.runs:
        run = json.load(open(f))
        rep = evaluate_run(run, cfg, att)
        rep["rmse"] = run.get("final_metrics", {}).get("rmse")
        reports.append(rep)
    cross = cross_run_checks(reports)
    paper_hits = scan_paper(open(a.paper, errors="ignore").read()) if a.paper else []
    os.makedirs(a.out, exist_ok=True)
    from src.utils.run_identity import save_json_timestamped
    save_json_timestamped(os.path.join(a.out, "sec_compliance_report.json"),
                          {"runs": reports, "cross_run": cross, "paper_claims": paper_hits})
    L = ["# SEC technical-control screen", "", "_Not legal advice or a compliance certification._", ""]
    for r in reports:
        L += [f"## {r['model']} ({r['run_id']}): {r['overall']}", "", "| ID | Control | Status | Evidence |", "|---|---|---|---|"]
        L += [f"| {c['id']} | {c['title']} | {c['status']} | {c['evidence']} |" for c in r["checks"]] + [""]
    if cross:
        L += ["## Cross-run", ""] + [f"- **{c['id']} {c['status']}**: {c['evidence']}" for c in cross] + [""]
    if a.paper:
        L += ["## Manuscript claims", ""] + ([f"- {h['why']}: …{h['text']}…" for h in paper_hits] or ["- none matched"])
    open(os.path.join(a.out, "sec_compliance_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if a.strict and any(c["status"] == "FAIL" for r in reports for c in r["checks"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
