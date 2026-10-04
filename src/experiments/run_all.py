"""
Master experiment runner.

Executes every baseline + ablation in the config, with:
  - per-model seeding (not just once at the top of the whole run)
  - real train/val/test data (loaded, not assumed to already exist as .npy)
  - structured logging (.log + .json + a row in results/all_results.csv) per run
  - real adversarial-attack and entanglement-tomography evaluation

Run with:  python -m src.experiments.run_all
See RUNNING.md for the full step-by-step, including expected runtime.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.baselines.classical_lstm import ClassicalLSTM
from src.baselines.classical_gan_llm import ClassicalGANLLM
from src.baselines.qgan_llm import QGANLLM
from src.baselines.qlstm_forecaster import QLSTMForecaster
from src.baselines.qaoa_llm import QAOALLM
from src.baselines.llm_forecaster import LLMForecaster
from src.evaluation.talis import TaLISConfig, compute_talis
from src.utils.config import load_config, merge_override
from src.utils.logging_utils import RunLogger, make_run_id
from src.utils.reproducibility import set_all_seeds
from src.utils.step_tracer import StepTracer
from src.utils.progress import progress_bar


def load_processed_split(processed_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    X_path = os.path.join(processed_dir, f"X_{split}.npy")
    y_path = os.path.join(processed_dir, f"y_{split}.npy")
    if not (os.path.isfile(X_path) and os.path.isfile(y_path)):
        raise FileNotFoundError(
            f"\n\nProcessed data not found at {X_path} / {y_path}.\n"
            f"Run `python -m scripts.prepare_data` first (see RUNNING.md, Step 2)."
        )
    return np.load(X_path), np.load(y_path)


def run_one_model(model, model_name: str, splits: dict, cfg: dict, seed: int,
                  reuse_checkpoint: bool = False):
    set_all_seeds(seed)
    run_id = make_run_id(model_name)
    log_cfg = cfg.get("logging", {})
    run_logger = RunLogger(
        run_id,
        log_dir=log_cfg.get("log_dir", "logs"),
        results_dir=log_cfg.get("results_dir", "results"),
    )
    run_logger.log_config(model.config)
    run_logger.info(f"Starting training run for {model_name} (run_id={run_id})")
    tracer = StepTracer(run_id, log_dir=log_cfg.get("log_dir", "logs"), logger=run_logger)

    ckpt_path = getattr(model, "checkpoint_path", None)
    t0 = time.time()
    if reuse_checkpoint and ckpt_path and os.path.isfile(ckpt_path):
        # Skip training entirely and evaluate an already-trained checkpoint.
        # Only models that declare a `checkpoint_path` (currently Classical
        # LSTM) support this; everything else trains as normal.
        run_logger.info(f"Reusing existing checkpoint {ckpt_path}; skipping training")
        model.build()
        model.load_model(ckpt_path)
        tracer.log_step("TRAIN", 1, (time.time() - t0) * 1000,
                        {"reused_checkpoint": True}, f"Reused checkpoint at {ckpt_path}")
    else:
        model.train(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"], run_logger=run_logger)
        tracer.log_step("TRAIN", 1, (time.time() - t0) * 1000,
                        {"reused_checkpoint": False, "n_train_rows": len(splits["X_train"])},
                        f"Trained {model_name} from scratch")

    attack_cfg = cfg["adversarial"]
    # last_input_prices: last observed close price per test sequence — needed
    # by the attack success metric's directional-flip definition. Assumes the
    # first feature column is (or is derived from) the last observed close;
    # adjust the index below if your feature ordering differs.
    last_prices = splits["X_test"][:, 0]

    t0 = time.time()
    metrics = model.evaluate(
        splits["X_test"], splits["y_test"],
        attack_cfg=attack_cfg, last_input_prices=last_prices,
    )
    tracer.log_step("EVALUATE", 2, (time.time() - t0) * 1000, metrics,
                    f"Evaluated {model_name} on {len(splits['X_test']):,} test rows")

    if hasattr(model, "measure_latency"):
        latency_report = model.measure_latency(splits["X_test"])
        metrics["latency_mean_ms"] = latency_report["mean_latency_ms"]
        metrics["latency_sd_ms"] = latency_report["sd_latency_ms"]
        metrics["latency_raw_samples_ms"] = latency_report["raw_samples_ms"]
        run_logger.info(
            f"Latency: mean={latency_report['mean_latency_ms']:.3f}ms "
            f"SD={latency_report['sd_latency_ms']:.3f}ms "
            f"(n={latency_report['n_repeats']} timed single-sample forward passes)"
        )

    # TaLIS (Threat Latency-Impact Scoring) -- see src/evaluation/talis.py's
    # module docstring for the formula and, importantly, for what it
    # substitutes (ASR for dollar-impact, inference latency for detection
    # latency) since neither is measured directly by this pipeline.
    talis_cfg = TaLISConfig(**cfg.get("talis", {}))
    metrics["talis_score"] = compute_talis(metrics.get("asr"), metrics.get("latency_mean_ms"), talis_cfg)

    # Passthrough for mediation_pipelines.py's M3->DV3 pathway (quantum
    # noise injection level mediating FPR) -- not wired into a full
    # mediation test yet (needs fpr_per_run collected across several
    # independent runs, not just this one config's value), but the
    # column needs to exist in results/all_results.csv before that's
    # possible at all, so it's saved now rather than added later.
    if "noise_strength" in model.config:
        metrics["noise_strength"] = model.config["noise_strength"]
    metrics["seed"] = seed

    run_logger.log_final_metrics(metrics)
    run_logger.append_to_results_csv(
        model_name,
        {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
    )
    # The full per-sample latency distribution (not just mean/SD) is kept in
    # the structured .json log via save_json() below, not the CSV -- this is
    # exactly what's needed to check a claim like "97% of runs fall below
    # 40ms" against the real distribution rather than assuming normality
    # from mean/SD alone.
    run_logger.save_json()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config YAML (default: config/default_config.yaml)")
    parser.add_argument("--only", default=None,
                         help="Comma-separated subset of model names to run (default: all)")
    parser.add_argument("--reuse-checkpoints", action="store_true",
                         help="For models that support it (Classical LSTM), skip training and "
                              "evaluate the existing checkpoint in models/.")
    parser.add_argument("--list-models", action="store_true",
                         help="Print the resolved model names (baselines + config-defined "
                              "ablations) as a JSON list to stdout and exit without running "
                              "anything. Used by run_pipeline.py to enumerate per-model steps "
                              "without duplicating the ablation-merging logic below.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Override config['seed'] for this run without editing the config "
                              "file. Added for scripts/run_repeated_experiment.py, which needs a "
                              "different seed per repeated run of the same model config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]

    models_to_run = {
        "Classical LSTM": lambda: ClassicalLSTM(cfg["classical_lstm"], seed=seed),
        "Classical GAN-LLM": lambda: ClassicalGANLLM(cfg["classical_gan_llm"], seed=seed),
        "QGAN-LLM": lambda: QGANLLM(cfg["qgan_llm"], seed=seed),
        "QLSTM Forecaster": lambda: QLSTMForecaster(cfg["qlstm_forecaster"], seed=seed),
        "QAOA-Enhanced LLM": lambda: QAOALLM(cfg["qaoa_llm"], seed=seed),
    }
    # LLM Forecaster: only registered when NVIDIA_API_KEY is actually set.
    # Matches nvidia_finetune.py's established philosophy -- no silent
    # fabrication -- but at the REGISTRATION level rather than making every
    # other user's run_pipeline.py invocation crash on a missing key for a
    # baseline they may not even want. Without the key, this model simply
    # doesn't appear in --list-models' output, so run_pipeline.py's
    # per-model steps never try to run it. With the key, it runs for real
    # (and fails loudly, per llm_inference.py, if the key turns out to be
    # invalid or the endpoint rejects the request).
    if os.environ.get("NVIDIA_API_KEY"):
        models_to_run["LLM Forecaster"] = lambda: LLMForecaster(cfg.get("llm_forecaster", {}), seed=seed)
    for ablation in cfg["ablations"]:
        ablation_cfg = merge_override(cfg["qgan_llm"], {k: v for k, v in ablation.items() if k != "name"})
        models_to_run[ablation["name"]] = (lambda c=ablation_cfg: QGANLLM(c, seed=seed))

    if args.only:
        selected = set(name.strip() for name in args.only.split(","))
        models_to_run = {k: v for k, v in models_to_run.items() if k in selected}

    if args.list_models:
        import json
        print(json.dumps(list(models_to_run.keys())))
        return

    processed_dir = cfg["data"]["processed_dir"]
    splits = {}
    for split in ("train", "val", "test"):
        X, y = load_processed_split(processed_dir, split)
        splits[f"X_{split}"] = X
        splits[f"y_{split}"] = y

    all_metrics = {}
    for name, factory in progress_bar(list(models_to_run.items()), total=len(models_to_run),
                                      desc="run_all.py models", unit="model"):
        print("=" * 70)
        print(f"Running: {name}")
        print("=" * 70)
        model = factory()
        metrics = run_one_model(model, name, splits, cfg, seed,
                                reuse_checkpoint=args.reuse_checkpoints)
        all_metrics[name] = metrics

    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE — summary:")
    print("=" * 70)
    for name, metrics in all_metrics.items():
        printable = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        print(f"{name}: {printable}")


if __name__ == "__main__":
    main()
