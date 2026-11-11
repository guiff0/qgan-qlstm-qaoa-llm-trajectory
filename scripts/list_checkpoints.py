"""Inventory of models/*.ckpt: fingerprint, completeness, entanglement trajectory, locks, synthetic-row caches, and (with
--identify) WHICH condition and seed each checkpoint belongs to.

    python -m scripts.list_checkpoints                       # models/ (no torch needed)
    python -m scripts.list_checkpoints --models-dir D:\\x\\models --json results\\checkpoints.json
    python -m scripts.list_checkpoints --csv results\\all_results.csv   # condition/seed from sidecars + models/history archives (no torch)
    python -m scripts.list_checkpoints --identify --write-sidecars    # IDENTIFY ALL: recompute fingerprints (needs torch + pennylane),
                                                                      # then write models/<ckpt>.id.json so the next listing needs no torch
    python -m scripts.list_checkpoints --identify --seeds 43 44 45 46 47 --conditions "QGAN-LLM" "QGAN-LLM (no noise)"

How a checkpoint is identified (src/utils/checkpoint_identity.py), strongest first: (1) sidecar models/<ckpt>.id.json, written by
every run since the Nov-11 code; (2) the archived copy models/history/<stem>__<run_id>.ckpt whose run_id names the condition, and the
results row with that run_id gives the seed; (3) --identify: recompute hash(config, seed) for every condition x seed. A checkpoint that
none of these identifies was written by a different model config/code: it will not resume, the run retrains.

What the columns mean
  state        COMPLETE = training finished AND the head is fitted (a re-run resumes it and goes straight to evaluation).
               INCOMPLETE (next epoch k) = training stopped after k epochs; a re-run continues from k.
  head         generator-path head fitted on [X, G(X)] (input dim 64 = 32 features + 32 generated) and its train MSE.
  entropy      mean entanglement entropy of the generator at epoch 4 ... last logged epoch (max possible shown); H2/H5 read this.
  lock         <ckpt>.train.lock = a process was training it. alive=False means the process is gone (crash/kill): the lock is
               STALE, the next run takes it over by itself (src/utils/io.py training_lock); you do not have to delete it.

The baseline arm (persistence + ridge on the inputs) has NO checkpoint by design: it is closed form. See
src/evaluation/arm_ledger.py and the THREE-ARM LEDGER that scripts.run_hypothesis_tests prints.
Read-only: it never writes to models/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.arm_ledger import checkpoint_inventory  # noqa: E402
from src.utils import checkpoint_identity as cid  # noqa: E402


def default_factories(cfg: dict) -> dict:
    """condition name -> callable(seed) -> model, mirroring src/experiments/run_all.py (needs torch/pennylane)."""
    from src.baselines.classical_gan_llm import ClassicalGANLLM
    from src.baselines.qgan_llm import QGANLLM
    from src.utils.config import merge_override
    f = {"QGAN-LLM": lambda s: QGANLLM(cfg["qgan_llm"], seed=s),
         "Classical GAN-LLM": lambda s: ClassicalGANLLM(cfg["classical_gan_llm"], seed=s)}
    for ab in cfg.get("ablations") or []:
        c = merge_override(cfg["qgan_llm"], {k: v for k, v in ab.items() if k != "name"})
        f[ab["name"]] = (lambda s, c=c: QGANLLM(c, seed=s))
    for ab in cfg.get("classical_ablations") or []:
        c = merge_override(cfg["classical_gan_llm"], {k: v for k, v in ab.items() if k != "name"})
        f[ab["name"]] = (lambda s, c=c: ClassicalGANLLM(c, seed=s))
    return f


def expected_fingerprints(factories: dict, conditions, seeds, fingerprint_fn) -> dict:
    """{fingerprint: (condition, seed)} for every condition x seed. A fingerprint hashes (model config, seed, tag)."""
    out = {}
    for name in conditions:
        if name not in factories:
            continue
        for s in seeds:
            m = factories[name](s)
            tag = "classical_gan_llm" if name.startswith("Classical") else "qgan_llm"
            fp = fingerprint_fn(m.config, s, tag)
            out[fp] = (name, s)
    return out


def identify(recs: list, mapping: dict) -> None:
    for r in recs:
        hit = mapping.get(r.get("fingerprint") or r.get("fingerprint_in_name"))
        r["condition"], r["seed"] = hit if hit else (None, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--json", default=None, help="also write the full inventory here")
    ap.add_argument("--identify", action="store_true", help="recompute fingerprints to find condition+seed (imports the model classes)")
    ap.add_argument("--csv", default=os.path.join("results", "all_results.csv"), help="gives the seed for archived run_ids")
    ap.add_argument("--write-sidecars", action="store_true", help="write models/<ckpt>.id.json for every identified checkpoint")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(0, 63)))
    ap.add_argument("--conditions", nargs="+", default=None, help="default: every condition in the config")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.models_dir):
        print(f"no such folder: {a.models_dir}")
        return 2
    inv = checkpoint_inventory(a.models_dir)
    recs = inv["checkpoints"]
    hist = cid.history_index(a.models_dir)
    runs = cid.load_run_table(a.csv if os.path.isabs(a.csv) else os.path.join(ROOT, a.csv))
    fp_map = None
    if a.identify:
        try:
            from src.utils.config import load_config
            from src.utils.gan_training import config_fingerprint
            cfg = load_config(a.config)
            fac = default_factories(cfg)
            fp_map = expected_fingerprints(fac, a.conditions or list(fac), a.seeds, config_fingerprint)
        except Exception as exc:  # noqa: BLE001
            print(f"--identify unavailable here ({type(exc).__name__}: {exc}); using sidecars and archives only.")
    for r in recs:
        r["identity"] = cid.identify_record(r, os.path.join(a.models_dir, r["file"]), hist, runs, fp_map)
    if not recs:
        print(f"no *.ckpt files in {a.models_dir}")
        return 1
    W = (27, 28, 22, 38, 17)
    print(f"{'file':<{W[0]}}{'state':<{W[1]}}{'head':<{W[2]}}{'entropy first -> last (max)':<{W[3]}}{'lock':<{W[4]}}{'MB':>5}  condition / seed (identified by)")
    for r in recs:
        if r["state"] == "UNREADABLE":
            print(f"{r['file']:<{W[0]}}UNREADABLE  {r.get('error')}")
            continue
        head = f"dim {r['head_input_dim']} mse {r['head_mse']:.2e}" if r["head_fitted"] and r["head_mse"] is not None else \
            ("not fitted yet" if not r["head_fitted"] else f"dim {r['head_input_dim']}")
        ent = (f"{r['entropy_first']:.3f} -> {r['entropy_last']:.3f} (ep {r['last_entropy_epoch']}, max {r['entropy_max_possible']:.0f})"
               if r["entropy_points"] else "no entropy logged")
        lk = r["lock"]
        lock = "-" if not lk else f"pid {lk['pid']} {'ALIVE' if lk['alive'] else 'stale'}"
        i = r["identity"]
        who = (f"{i['condition']} / {i['seed'] if i['seed'] is not None else '?'}  ({i['by']})" if i.get("by") else "UNIDENTIFIED")
        flag = "" if r["fingerprint_matches_name"] else "  !name!=embedded fingerprint"
        print(f"{r['file']:<{W[0]}}{r['state']:<{W[1]}}{head:<{W[2]}}{ent:<{W[3]}}{lock:<{W[4]}}{r['bytes'] / 1e6:>5.1f}  {who}{flag}")
    ok = [r for r in recs if r.get("identity", {}).get("by")]
    by_src = {}
    for r in ok:
        by_src[r["identity"]["by"]] = by_src.get(r["identity"]["by"], 0) + 1
    print(f"\nIDENTIFIED {len(ok)} of {len(recs)}: " + (", ".join(f"{n} by {k}" for k, n in by_src.items()) or "none"))
    for r in recs:
        i = r.get("identity", {})
        if not i.get("by") and r["state"] != "UNREADABLE":
            print(f"  UNIDENTIFIED {r['file']}: {i.get('note')}")
        elif i.get("also_used_by"):
            print(f"  {r['file']} is shared by several conditions: {i['also_used_by']}")
        elif i.get("note"):
            print(f"  {r['file']}: {i['note']}")
    if a.write_sidecars:
        n = 0
        for r in ok:
            i = r["identity"]
            if i["by"] == "sidecar" or not i.get("condition"):
                continue
            sp = cid.sidecar_path(os.path.join(a.models_dir, r["file"]))
            if os.path.exists(sp):
                continue
            with open(sp, "w", encoding="utf8") as fh:
                json.dump({"condition": i["condition"], "seed": i["seed"], "source": f"backfilled from {i['by']}",
                           "first_saved_utc": r["mtime_utc"]}, fh, indent=2)
            n += 1
        print(f"sidecars written: {n} (never overwrites an existing one)")
    print("\nSUMMARY BY PREFIX")
    for p, b in inv["by_prefix"].items():
        print(f"  {p}: {b['complete']} complete, {b['incomplete']} incomplete, {b['unreadable']} unreadable, "
              f"{b['stale_lock']} stale lock(s), {b['live_lock']} live lock(s)")
    caches = [(r["file"], r["synth_cache"]) for r in recs if r.get("synth_cache")]
    if caches:
        print(f"  synthetic-row caches next to checkpoints: {len(caches)} "
              f"({caches[0][1]['rows']:,} rows x {caches[0][1]['dims']} dims each)")
    have = {r["file"][:-len(".ckpt")] for r in recs}
    orphans = sorted(f for f in os.listdir(a.models_dir) if f.endswith("_synth.npy") and f[:-len("_synth.npy")] not in have)
    if orphans:
        print(f"  synthetic-row caches WITHOUT a checkpoint here ({len(orphans)}): {', '.join(orphans)} "
              "(the checkpoint was moved or deleted; the cache alone cannot resume training)")
    print("\nNOT HERE BY DESIGN: the baseline arm (persistence + ridge on the inputs) is closed form, so it has no checkpoint; "
          "its results are in all_results.csv (persistence_rmse) and results/feature_map_advantage_q*.json (arm 'linear').")
    if any(r.get("lock") and not r["lock"]["alive"] for r in recs):
        print("Stale locks are harmless: the next run takes them over. Lock liveness is checked on THIS machine only.")
    if a.json:
        with open(a.json, "w", encoding="utf8") as fh:
            json.dump(inv, fh, indent=2, default=str)
        print(f"written: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
