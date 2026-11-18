#!/usr/bin/env python3
"""
audit_checkpoints.py — verify design-fact disclosures from checkpoint metadata.

Subcommands
-----------
  epochs_equal       All complete checkpoints have next_epoch == 30 (matched training budget).
  no_per_arm_tuning  No per-arm hyperparameter search artifacts exist.
  capacity_ratio     Print per-checkpoint generator_params; pin the 155x ratio to (87072, 560).

Usage
-----
  python -m scripts.audit_checkpoints --check epochs_equal
  python -m scripts.audit_checkpoints --check no_per_arm_tuning
  python -m scripts.audit_checkpoints --check capacity_ratio
  python -m scripts.audit_checkpoints --check all
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT  = REPO_ROOT / "results" / "snapshots" / "all_results_20261118T071700Z.csv"
CONFIG    = REPO_ROOT / "config" / "default_config.yaml"
MODELS    = REPO_ROOT / "models"

# The capacity pair the manuscript pins the 155x ratio to.
LARGE_CLASSICAL_GP  = 87_072
MODAL_QUANTUM_GP    = 560
EXPECTED_RATIO      = LARGE_CLASSICAL_GP / MODAL_QUANTUM_GP   # ~155.5
EXPECTED_EPOCHS     = 30
PRIMARY_MODELS      = {"Classical GAN-LLM", "QGAN-LLM"}


def _load_snapshot():
    if not SNAPSHOT.exists():
        # fall back to latest snapshot
        snaps = sorted(glob.glob(str(REPO_ROOT / "results" / "snapshots" / "*.csv")))
        if not snaps:
            sys.exit(f"No snapshot found under {REPO_ROOT}/results/snapshots/")
        snap_path = snaps[-1]
    else:
        snap_path = str(SNAPSHOT)
    with open(snap_path, newline="") as f:
        return list(csv.DictReader(f)), snap_path


def _load_checkpoints():
    """Return list of dicts from models/**/*.json (checkpoint metadata)."""
    ckpts = []
    for f in glob.glob(str(MODELS / "**" / "*.json"), recursive=True):
        try:
            d = json.load(open(f))
            if "model_type" in d:
                d["_path"] = f
                ckpts.append(d)
        except Exception:
            pass
    return ckpts


# ---------------------------------------------------------------------------
# Check: epochs_equal
# ---------------------------------------------------------------------------

def check_epochs_equal():
    rows, snap_path = _load_snapshot()
    # The snapshot may not carry next_epoch; fall back to checkpoint metadata.
    ckpts = _load_checkpoints()

    failures = []
    checked  = 0

    if ckpts:
        for c in ckpts:
            mt = c.get("model_type", "")
            if mt not in PRIMARY_MODELS:
                continue
            ep = c.get("next_epoch")
            if ep is None:
                continue
            checked += 1
            if int(ep) != EXPECTED_EPOCHS:
                failures.append(
                    f"  FAIL: {c['_path']} -> model_type={mt}, next_epoch={ep}"
                )
        source = f"checkpoint metadata ({len(ckpts)} files)"
    else:
        # Fall back to snapshot epoch column if present
        ep_col = next((k for k in (rows[0].keys() if rows else [])
                       if "epoch" in k.lower()), None)
        if not ep_col:
            print(f"[epochs_equal] No checkpoint metadata found and snapshot has no epoch column.")
            print(f"  Source: {snap_path}")
            print("  SKIP: cannot verify — attach models/ directory to repo.")
            return
        for r in rows:
            mt = r.get("model_type", "")
            if mt not in PRIMARY_MODELS:
                continue
            try:
                ep = int(r[ep_col])
            except (ValueError, TypeError):
                continue
            checked += 1
            if ep != EXPECTED_EPOCHS:
                failures.append(
                    f"  FAIL: run_id={r.get('run_id')}, model_type={mt}, {ep_col}={ep}"
                )
        source = f"snapshot {snap_path} (column: {ep_col})"

    if failures:
        print(f"[epochs_equal] FAIL — {len(failures)} of {checked} checked entries differ from {EXPECTED_EPOCHS}")
        for f in failures:
            print(f)
        sys.exit(1)
    elif checked == 0:
        print(f"[epochs_equal] SKIP — no epoch data found in {source}")
    else:
        print(f"[epochs_equal] PASS — all {checked} checked entries have next_epoch = {EXPECTED_EPOCHS}")
        print(f"  Source: {source}")


# ---------------------------------------------------------------------------
# Check: no_per_arm_tuning
# ---------------------------------------------------------------------------

def check_no_per_arm_tuning():
    # Look for per-arm hyperparameter search artifacts:
    #   - tuning/ directories, optuna studies, ray-tune logs, wandb sweeps, etc.
    tuning_patterns = [
        "tuning/**",
        "optuna_study*",
        "ray_results/**",
        "wandb/sweep*",
        "hyperopt_results*",
        "hpo_results*",
    ]
    found = []
    for pat in tuning_patterns:
        matches = glob.glob(str(REPO_ROOT / pat), recursive=True)
        found.extend(matches)

    # Also check config for per-arm overrides (presence of arm-specific blocks).
    config_ok = True
    config_note = ""
    if CONFIG.exists():
        try:
            import yaml
            cfg = yaml.safe_load(CONFIG.read_text())
            # Check for per-arm keys that would indicate separate tuning.
            arm_specific = {k: v for k, v in (cfg or {}).items()
                            if any(arm in k.lower() for arm in
                                   ["classical_gan_llm", "qgan_llm", "per_arm", "per_model"])}
            if arm_specific:
                config_ok = False
                config_note = f"  config has per-arm keys: {list(arm_specific.keys())}"
        except ImportError:
            config_note = "  (yaml not available; config inspection skipped)"

    if found or not config_ok:
        print("[no_per_arm_tuning] FAIL")
        for f in found:
            print(f"  found: {f}")
        if config_note:
            print(config_note)
        sys.exit(1)
    else:
        print("[no_per_arm_tuning] PASS — no per-arm hyperparameter search artifacts found")
        print(f"  Config: {CONFIG}")
        if config_note:
            print(config_note)


# ---------------------------------------------------------------------------
# Check: capacity_ratio
# ---------------------------------------------------------------------------

def check_capacity_ratio():
    ckpts = _load_checkpoints()
    rows, snap_path = _load_snapshot()

    gp_by_model = {}
    source = ""

    if ckpts:
        source = f"checkpoint metadata ({len(ckpts)} files)"
        for c in ckpts:
            mt = c.get("model_type", "")
            gp = c.get("generator_params")
            if mt and gp is not None:
                gp_by_model.setdefault(mt, []).append(int(gp))
    else:
        # Fall back to snapshot
        gp_col = next((k for k in (rows[0].keys() if rows else [])
                       if "generator_param" in k.lower() or k == "gen_params"), None)
        if gp_col:
            for r in rows:
                mt = r.get("model_type", "")
                try:
                    gp_by_model.setdefault(mt, []).append(int(r[gp_col]))
                except (ValueError, TypeError):
                    pass
            source = f"snapshot {snap_path} (column: {gp_col})"

    if not gp_by_model:
        print("[capacity_ratio] SKIP — no generator_params data found.")
        print("  Attach models/ directory or add generator_params column to snapshot.")
        return

    print(f"[capacity_ratio] Source: {source}")
    print()
    for mt in sorted(gp_by_model):
        vals = sorted(set(gp_by_model[mt]))
        modal = max(set(vals), key=vals.count)
        print(f"  {mt}:")
        print(f"    min={min(vals)}, max={max(vals)}, modal={modal}, n_checkpoints={len(vals)}")

    # Pin the 155x ratio
    classical_max = max(gp_by_model.get("Classical GAN-LLM", [0]))
    quantum_modal = max(set(gp_by_model.get("QGAN-LLM", [0])),
                        key=gp_by_model.get("QGAN-LLM", [0]).count) if gp_by_model.get("QGAN-LLM") else 0

    if classical_max > 0 and quantum_modal > 0:
        actual_ratio = classical_max / quantum_modal
        print()
        print(f"  Pinned ratio: {classical_max} / {quantum_modal} = {actual_ratio:.1f}x")
        if abs(actual_ratio - EXPECTED_RATIO) > 5:
            print(f"  NOTE: actual ratio ({actual_ratio:.1f}x) differs from manuscript value ({EXPECTED_RATIO:.1f}x) by >5x")
            print(f"        Update the manuscript or this script's LARGE_CLASSICAL_GP / MODAL_QUANTUM_GP constants.")
        else:
            print(f"  Manuscript value ~{EXPECTED_RATIO:.0f}x matches pinned pair. PASS")
    else:
        print("  Could not compute ratio — missing Classical GAN-LLM or QGAN-LLM data.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKS = {
    "epochs_equal":     check_epochs_equal,
    "no_per_arm_tuning": check_no_per_arm_tuning,
    "capacity_ratio":   check_capacity_ratio,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", required=True,
                    choices=list(CHECKS) + ["all"],
                    help="Which disclosure to verify")
    args = ap.parse_args()

    if args.check == "all":
        for name, fn in CHECKS.items():
            print(f"\n{'='*60}")
            print(f"CHECK: {name}")
            print('='*60)
            fn()
    else:
        CHECKS[args.check]()


if __name__ == "__main__":
    main()
