# FIX_NOTES_OCT28.md -- "insufficient data", analysis re-runs, LLM-probe crashes

Unzip over the repo root (overwrites 5 files, adds 1 test + this note). Do NOT use --reset.

## What was wrong
1. "insufficient data": scripts/run_hypothesis_tests.py collapsed results to ONE row per config
   (latest_per_config), and every config had been run once, so each t-test had n=1 per group.
   It now counts runs per (config, seed): re-evaluating one checkpoint is not a new run, a new
   seed is.
2. Tests now degrade instead of stopping: n>=2 vs n>=2 -> Welch t-test ("ok"); one side n=1 ->
   Crawford-Howell single-case t-test ("exploratory_single_case"); otherwise a structured
   "insufficient_data" with the exact command to run. H4 reports "degenerate_by_design" with the
   reason (the GAN forecast head is a closed-form linear fit on real rows; the generator is not on
   the forecast path, so RMSE/ASR are identical for every GAN model and seed).
3. Classical LSTM and QGAN-LLM failed after 3m22s: run_all.py probed the NVIDIA LLM endpoint
   before training and died on a ReadTimeout, although Classical LSTM never uses the LLM. The probe
   now only runs when a selected model uses the LLM, retries 3x (LLM_PROBE_RETRIES), and is fatal
   only for models that REQUIRE the LLM (LLM Forecaster, QLSTM-LLM, QAOA-LLM, Classical-Features-LLM).

## New commands
    python -m scripts.run_hypothesis_tests            # standalone, any time, never needs the pipeline
    python run_pipeline.py --stats-only               # same, through the pipeline's state/logging
    python run_pipeline.py --analysis-only            # verify_chapter4 + run_hypothesis_tests
    python run_pipeline.py --steps run_hypothesis_tests,scan_logs_for_pii
    python run_pipeline.py --yes --extra-seeds 1 --seed-models "Classical GAN-LLM,QGAN-LLM (no noise)"

verify_chapter4 and run_hypothesis_tests (the "analysis" steps) re-run on EVERY invocation, ignore
--yes / --no-skip / the prompt, are not blocked by failed training or required-step failures, and
a failure in one does not stop the other. Failed train_eval steps are never skipped.

## Minimum extra runs to turn the current results into real two-sample tests
    Classical GAN-LLM            +1 seed   ~20 min   -> H1 (fidelity) n=2 vs n=4
    QGAN-LLM (no noise)          +1 seed   ~12.4 h   -> H3 (poisoning) n=2 vs n=3
H4 (quantum vs classical RMSE/ASR) cannot be fixed by seeds: see "degenerate_by_design".
