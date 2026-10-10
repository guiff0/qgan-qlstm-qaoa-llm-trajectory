# Fix patch: Classical GAN-LLM crash, QGAN-LLM stall, run time

Unzip over the repo root (overwrites 8 files, adds 3 + 1 test). Do NOT run
`run_pipeline.py --reset`: data/pipeline_state.json already marks data
acquisition, prepare_data and Classical LSTM as done, and a plain
`python run_pipeline.py` resumes at Classical GAN-LLM.

Then: `python run_pipeline.py` (answer "skip completed" if asked).
If a GAN run is interrupted, re-run the same command: it resumes from
models/*.ckpt at the last finished epoch.

Changed: src/baselines/qlstm_forecaster.py, src/evaluation/metrics.py, src/evaluation/poisoning_resistance.py,
src/quantum/circuits.py, src/baselines/classical_gan_llm.py,
src/baselines/qgan_llm.py, config/default_config.yaml
Added: src/quantum/pqc_runner.py, src/utils/gan_training.py, tests/test_pqc_runner.py
