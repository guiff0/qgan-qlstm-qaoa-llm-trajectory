"""
Rebuild / extend results/history/all_results_ledger.csv from the structured run logs.

Every run already leaves logs/<run_id>.json (config, environment, final metrics). That is enough to
reconstruct a results row even if results/all_results.csv was reset, filtered or overwritten. Rows for
run_ids that are already in the ledger are left alone, so this is safe to run any time (idempotent).

    python -m scripts.rebuild_ledger [--logs logs] [--results results]
"""
import argparse, csv, glob, json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.run_identity import append_ledger, archive_file, config_of_run_id, utc_stamp


def main(logs="logs", results="results"):
    ledger = os.path.join(results, "history", "all_results_ledger.csv")
    have = set()
    if os.path.isfile(ledger):
        with open(ledger, newline="") as f:
            have = {r["run_id"] for r in csv.DictReader(f)}
    # keep the current aggregate files before anything else touches them
    for fn in ("all_results.csv", "chapter4_numbers.json", "hypothesis_test_results.json"):
        archive_file(os.path.join(results, fn), tag="backfill_" + utc_stamp())
    added = 0
    for path in sorted(glob.glob(os.path.join(logs, "*.json"))):
        rec = json.load(open(path))
        rid = rec.get("run_id") or os.path.basename(path)[:-5]
        if rid in have or "final_metrics" not in rec:
            continue
        env = rec.get("environment", {})
        row = {"run_id": rid, "model_type": rec.get("final_metrics", {}).get("model_type") or config_of_run_id(rid),
               "pipeline_run_id": env.get("pipeline_run_id"), "run_utc": env.get("timestamp_utc"),
               "git_commit": env.get("git_commit"), "source": "rebuilt_from_log"}
        row.update({k: v for k, v in rec["final_metrics"].items() if isinstance(v, (int, float, str, bool)) or v is None})
        append_ledger(results, row); have.add(rid); added += 1
    print(f"ledger: {len(have)} runs ({added} added) -> {ledger}")
    return added


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--logs", default="logs"); ap.add_argument("--results", default="results")
    a = ap.parse_args(); main(a.logs, a.results)
