"""Run-identity helpers shared by the pipeline, collectors and tests.

run_id = "<slug of model name>_<UTC timestamp>" (see logging_utils.make_run_id).
All QGAN-LLM ablation configurations are written to results/all_results.csv
with model_type == "QGAN-LLM", so the configuration can only be recovered from
the run_id slug. Matching on model_type alone made run_pipeline.py report the
three ablation configs as failed on every run and re-run them each time.
"""
from __future__ import annotations

import re

_TS = re.compile(r"_\d{8}T\d{6}Z$")


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def config_of_run_id(run_id: str) -> str:
    return _TS.sub("", str(run_id))


def latest_per_config(df):
    """One row per configuration (the newest run_id), so repeated evaluations of
    the same checkpoint are not counted as independent observations."""
    out = df.copy()
    out["config"] = out["run_id"].map(config_of_run_id)
    out["_ts"] = out["run_id"].astype(str).str.extract(r"(\d{8}T\d{6}Z)$")[0]
    out = out.sort_values("_ts").drop_duplicates(subset="config", keep="last")
    return out.drop(columns="_ts").reset_index(drop=True)
