# Run history: dated results that later runs cannot erase

**Ids.** `run_id` = `<model slug>_<UTC timestamp>` (one per model run; bumped by one second if the id is taken, so a log is never overwritten). `pipeline_run_id` = `P<UTC timestamp>-<git sha>` (one per `run_pipeline.py` call; shared with every subprocess through the `PIPELINE_RUN_ID` environment variable). Both are written into `logs/<run_id>.json` (`environment.pipeline_run_id`, `config_sha256`, `git_commit`) and into every results row (`pipeline_run_id`, `run_utc`, `git_commit`).

| File that is rewritten in place | Where the old versions are kept |
|---|---|
| `results/all_results.csv` | `results/history/all_results_ledger.csv` (append-only, one row per run) |
| `results/chapter4_numbers.json` | `results/history/chapter4_numbers__<pipeline_run_id>.json` |
| `results/hypothesis_test_results.json` | `results/history/hypothesis_test_results__<pipeline_run_id>[_final].json` (also records `pipeline_run_id`) |
| `data/pipeline_state.json` (also on `--reset`) | `data/history/pipeline_state__<pipeline_run_id>.json` |
| `models/classical_lstm_best.pt` and other checkpoints | `models/history/<name>__<run_id>.pt` |
| `data/processed/*.npy` | `data/processed/manifest__<UTC stamp>.json` (sha256 and size of every array, config hash) |

Identical files are not copied twice; an existing archive name is never overwritten (a numeric suffix is added). If `all_results.csv` is ever reset or filtered, rebuild the ledger from the logs: `python -m scripts.rebuild_ledger`.

## Time-stamped JSON results

Every generated JSON result is written twice by `save_json_timestamped` (`src/utils/run_identity.py`): an immutable `<name>__<UTC YYYYMMDDTHHMMSSZ>.json` named by the moment it was generated, and the stable `<name>.json` as the latest copy, so existing readers keep working. A clash in the same second gets `_1`, `_2`; NaN is written as `null`. A pre-existing un-versioned file is copied to `history/` the first time.

| Stable (latest) file | Versioned files |
|---|---|
| `results/chapter4_numbers.json` | `results/chapter4_numbers__<stamp>.json` |
| `results/hypothesis_test_results.json` (also holds `generated_utc`) | `results/hypothesis_test_results__<stamp>.json` |
| `results/no_advantage_diagnosis.json` | `results/no_advantage_diagnosis__<stamp>.json` |
| `results/qpca_results.json` | `results/qpca_results__<stamp>.json` |
| `results/sec_compliance_report.json` | `results/sec_compliance_report__<stamp>.json` |
| `logs/pipeline/pipeline_run.json` (one record per `run_pipeline.py` call: status, start/finish, per-step state) | `logs/pipeline/pipeline_run__<stamp>.json` |

List versions in code with `list_versions(path)` (oldest first).
