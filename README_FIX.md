# Fix patch: Classical GAN-LLM crash, QGAN-LLM stall, run time

Unzip over the repo root (overwrites 8 files, adds 3 + 1 test). Do NOT run
`run\_pipeline.py --reset`: data/pipeline\_state.json already marks data
acquisition, prepare\_data and Classical LSTM as done, and a plain
`python run\_pipeline.py` resumes at Classical GAN-LLM.

Then: `python run\_pipeline.py` (answer "skip completed" if asked).
If a GAN run is interrupted, re-run the same command: it resumes from
models/\*.ckpt at the last finished epoch.

Changed: src/baselines/classical\_lstm.py (adds ASR so TaLIS is defined), src/baselines/qlstm\_forecaster.py, src/evaluation/metrics.py, src/evaluation/poisoning\_resistance.py,
src/quantum/circuits.py, src/baselines/classical\_gan\_llm.py,
src/baselines/qgan\_llm.py, config/default\_config.yaml
Added: src/utils/stage\_log.py (timestamps), src/quantum/pqc\_runner.py, src/utils/gan\_training.py, tests/test\_pqc\_runner.py





Yes, full is 40 runs and minimum is 12 runs.





.EXAMPLE   .\\scripts\\run\_nov14.ps1

.EXAMPLE   .\\scripts\\run\_nov14.ps1 -SkipPilots                    # study only

.EXAMPLE   .\\scripts\\run\_nov14.ps1 -PilotsOnly                    # pilots and table only, no calibration, no study

.EXAMPLE   .\\scripts\\run\_nov14.ps1 -AllowUncalibrated             # if step 3 finds no stable strength: run anyway, H4/H5 rows are flagged

.EXAMPLE   .\\scripts\\run\_nov14.ps1 -NoRefreeze                    # keep the old frozen record; changed files are then listed as deviations

.EXAMPLE   .\\scripts\\run\_nov14.ps1 -BaseSeed 53 -Workers 4 -Deviation "my own wording"

.\\scripts\\resume\_study.ps1

&#x09;Runs	GAN seeds	Command

Full	40	10 (53–62)	.\\scripts\\run\_full\_study.ps1 --new-experiment --base-seed 53 --workers 4 --record-deviation "..."

Minimum	12	3 (53–55)	.\\scripts\\run\_full\_study.ps1 --three-arm-min --new-experiment --base-seed 53 --n-runs 3 --workers 4 --record-deviation "..."

