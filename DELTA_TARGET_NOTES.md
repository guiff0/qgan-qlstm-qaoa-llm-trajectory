# DELTA_TARGET_NOTES.md -- forecast the next-close CHANGE instead of the price level

Unzip over the repo root (after fix_oct28_pipeline_stats.zip). Do NOT use --reset.

## What changed
- `forecast_target: delta` (new, in config/default_config.yaml for classical_gan_llm, qgan_llm,
  qlstm_forecaster). Target = y[k+1] - y[k]; forecast = last close + head(X). A head that learns
  nothing predicts 0 and scores EXACTLY persistence. RMSE in this frame equals the price-forecast
  RMSE, so it is directly comparable with `persistence RMSE` and with old level-frame rows.
  Set `forecast_target: level` to get the old behaviour back.
- Adversarial "directional flip" is measured on the forecast change (reference 0) in the delta frame.
- Results gain a `target_frame` column; older rows have it empty (= level).

## What you do NOT have to retrain
- Classical GAN-LLM / QGAN-LLM: the generator never depends on the head. `forecast_target` is excluded
  from the checkpoint fingerprint. A finished level-frame checkpoint is archived to models/history/
  (tag `level_frame`), its head is refitted in closed form (minutes), and the checkpoint is rewritten.
  The pipeline skips a train_eval step whose result row already exists, so force these explicitly
  (explicit --steps never skip; the generator is not retrained, only the head is refitted):

      python run_pipeline.py --steps "train_eval:Classical GAN-LLM,train_eval:QGAN-LLM,train_eval:QGAN-LLM (12-qubit),train_eval:QGAN-LLM (linear topology),train_eval:QGAN-LLM (no noise)"
      python run_pipeline.py --analysis-only

  Evaluation itself still costs time (QGAN fidelity/resilience stages), but not the 7-13 h of training.

## What DOES retrain
- QLSTM Forecaster: the circuit is supervised on the target, so it cannot reuse a level-frame
  checkpoint. It gets a separate fingerprint in the delta frame (`qlstm_forecaster:delta`); the old
  1480-minute level-frame checkpoint is left untouched on disk. Expect another full QLSTM training run.
- QLSTM-LLM trains its own inner QLSTM, so it also retrains (force it with `--steps "train_eval:QLSTM Forecaster,train_eval:QLSTM-LLM"`); its prompt now says "next-bar change".

## Not changed (deliberately)
- Classical LSTM and QAOA-Enhanced LLM still regress the level, so they are not on the same footing
  as the three models above. Say the word and I will convert them too.

## Caveat found while testing (tests/test_delta_target.py)
Synthetic check: a level-frame linear head is 10-50x persistence when train/test price levels differ;
the delta head is ~1-2x. It is NOT guaranteed <= persistence: any feature that carries the price level
still leaks the train->test level shift through its coefficient. If delta RMSE is still above
persistence on the real data, the next fix is stationarising those features (difference/return them or
drop level-bearing PCA inputs), not a different generator.
