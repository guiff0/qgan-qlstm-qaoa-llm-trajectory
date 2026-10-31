"""
Explain, from artifacts a finished pipeline leaves behind, why no quantum
advantage is visible. Reads results/all_results.csv, models/*.synth.npy and
config/default_config.yaml; writes results/no_advantage_diagnosis.json.

Usage:  python -m scripts.diagnose_no_advantage
Needs only numpy/pandas/pyyaml (no torch, no PennyLane).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.advantage_diagnostics import (  # noqa: E402
    effective_rank, find_degenerate_groups, generator_rank_bound, lstm_generator_params,
    numerical_rank, qgan_generator_params, training_budget_fraction,
)
from src.utils.run_identity import config_of_run_id  # noqa: E402

N_TRAIN_ROWS = 3_715_921  # Chapter 3/4 training split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--config", default="config/default_config.yaml")
    ap.add_argument("--out", default="results/no_advantage_diagnosis.json")
    ap.add_argument("--n-train", type=int, default=N_TRAIN_ROWS)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    df = pd.read_csv(a.results)
    out: dict = {}

    # 1. identical forecasts / ASR across model families
    gan = df[df["model_type"].isin(["QGAN-LLM", "Classical GAN-LLM"])]
    out["C1_forecast_path"] = {
        "rmse_degenerate": find_degenerate_groups(
            {m: g["rmse"].dropna().to_numpy() for m, g in gan.groupby("model_type")}),
        "n_distinct_rmse_values_across_GAN_runs": int(gan["rmse"].round(12).nunique()),
        "explanation": ("forecast_head is fitted by closed-form least squares on REAL rows "
                        "(utils/gan_training.fit_linear_head_closed_form) and is the only module "
                        "used at inference; the generator never reaches it."),
    }

    # 2. generator output rank vs what is needed
    q = cfg["qgan_llm"]
    NF = cfg["model"]["n_features"]
    ranks = []
    for f in sorted(glob.glob(os.path.join(a.models_dir, "*.synth.npy"))):
        S = np.load(f)
        ranks.append({"file": os.path.basename(f), "shape": list(S.shape),
                      "numerical_rank": numerical_rank(S), "effective_rank": round(effective_rank(S), 2),
                      "per_feature_sd_min": float(S.std(0).min()), "per_feature_sd_max": float(S.std(0).max())})
    out["C2_generator_rank"] = {
        "n_features": NF,
        "rank_bound_20q_z": generator_rank_bound(20, NF),
        "rank_bound_12q_z": generator_rank_bound(12, NF),
        "observed": ranks,
    }

    # 3. unmatched hyper-parameters
    cg = cfg["classical_gan_llm"]
    out["C3_loss_weight"] = {
        "qgan_alpha_in_current_config": q.get("alpha"),
        "classical_mse_weight_in_current_config": cg.get("mse_weight", 0.1),
        "legacy_ratio_alpha_over_mse_weight": 10.0 / 0.1,
        "note": "runs in all_results.csv used alpha=10.0 (QGAN) vs 0.1 (classical).",
    }

    # 4. capacity
    out["C4_capacity"] = {
        "classical_generator_params": lstm_generator_params(cg["latent_dim"], cg["generator_hidden"], NF),
        "qgan_20q_params": qgan_generator_params(q["n_qubits"], q["n_layers"], NF),
        "qgan_12q_params": qgan_generator_params(12, q["n_layers"], NF),
    }
    c = out["C4_capacity"]
    c["ratio_classical_over_qgan_20q"] = round(c["classical_generator_params"] / c["qgan_20q_params"], 1)

    # 5. training budget actually used
    out["C5_budget"] = {
        "gan": training_budget_fraction(q.get("max_batches_per_epoch"), q["batch_size"], q["epochs"], a.n_train),
        "classical_lstm_for_contrast": "full pass, early stopping (12-19 epochs over 3.7M rows)",
    }

    # 6. replication
    runs = df.assign(config=df["run_id"].map(config_of_run_id))
    out["C6_replication"] = {
        "n_seeds_in_results": int(df["seed"].nunique()),
        "independent_configs": int(runs["config"].nunique()),
        "note": "one seed -> no variance estimate for any quantum-vs-classical contrast",
    }

    # 7. NISQ / simulation
    out["C7_simulation"] = {
        "statement": "All circuits are classically simulated; wall-clock speed-up is impossible by construction. "
                     "The only testable question is whether the circuit's function class helps at equal capacity.",
    }

    from src.utils.run_identity import save_json_timestamped
    save_json_timestamped(a.out, out)      # results/no_advantage_diagnosis__<UTC stamp>.json + latest
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
