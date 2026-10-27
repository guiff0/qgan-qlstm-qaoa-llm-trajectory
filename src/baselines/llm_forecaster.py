"""
LLM Forecaster: the first baseline in this pipeline whose forecast
actually comes from an LLM, rather than a classical nn.Linear head
dressed up with an LLM-sounding name. Every other "LLM" model
(Classical GAN-LLM, QGAN-LLM) names a `forecast_head` that is a plain
linear layer trained on real features -- src/llm/nvidia_finetune.py
existed as a real, working fine-tuning client, but nothing ever called
it or used an LLM's output as a prediction.

SCOPE, read before using this for real:
  - ZERO-SHOT / FEW-SHOT, not fine-tuned, by default. Fine-tuning is a
    separate, hours-long, asynchronous job
    (nvidia_finetune.submit_finetune_job + poll_finetune_status) that
    doesn't fit this codebase's synchronous train()/evaluate() contract
    -- train() here does NOT submit or wait for a fine-tuning job. Run
    that manually, get a fine-tuned model_id, then pass it as
    config["model_id"] to use it here, instead of a base model.
  - Input mismatch with every other baseline: this model needs actual
    recent CLOSE PRICES to build a sensible text prompt, but
    evaluate()'s X_test argument is PCA-transformed features, not
    interpretable prices. Reuses the SAME convention
    src/experiments/run_all.py already applies for last_input_prices
    (X[:, 0] as a stand-in for "the last observed close") -- an
    existing, documented proxy in this codebase, not a new assumption
    invented here, but still a proxy, not real price history.
  - Each prediction is a REAL network call (see src/llm/llm_inference.py)
    -- evaluating on the full ~750k-row test set would mean ~750k live
    API calls. This model only evaluates on a small subsample
    (config["max_llm_calls"], default 20) and reports metrics computed
    over just that subsample, with n_llm_samples recorded explicitly so
    it is never confused with the full-test-set metrics every other
    baseline in this pipeline reports.
  - UNVERIFIED AGAINST A LIVE ENDPOINT: see llm_inference.py's module
    docstring -- this sandbox has no NVIDIA_API_KEY and no network
    route to NVIDIA's API at all. Verify against a real account before
    trusting any number this produces.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .base import BaseForecastingModel
from ..evaluation.one_step_ahead import shift_for_one_step_ahead


class LLMForecaster(BaseForecastingModel):
    def __init__(self, config: Dict = None, seed: int = 42):
        default_config = {
            "model_id": "nvidia/llama-3.1-nemotron-70b-instruct",  # Llama 3.3 models are no longer served (Oct 2026)
            "api_base": "https://integrate.api.nvidia.com/v1",
            "window_size": 10,   # how many recent closes go into the prompt
            "horizon_minutes": 1,
            "n_windows": 102,    # stratified sample; each window costs 2 live calls
            "chat_fn": None,      # dependency injection point for tests -- see evaluate()
        }
        cfg = {**default_config, **(config or {})}
        super().__init__("LLM Forecaster", cfg)
        self.seed = seed

    def build(self):
        pass  # nothing to build locally -- inference happens against a hosted model

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        # Deliberately a no-op -- see module docstring. Fine-tuning (if
        # you want it) is run manually via nvidia_finetune.py, separately
        # from this pipeline's synchronous train() contract.
        self.is_trained = True

    def predict(self, X):
        raise NotImplementedError(
            "LLMForecaster.predict(X) isn't implemented standalone -- unlike every other "
            "baseline, it needs a raw CLOSE-PRICE window to build a prompt, not just a "
            "feature matrix. Use evaluate(), which builds that window itself from X[:, 0] "
            "(see module docstring)."
        )

    def evaluate(self, X_test, y_test, attack_cfg: Dict = None, last_input_prices=None, **kwargs):
        """Control condition of the LLM pipeline: the same dual-output prompt, windows, attacks and
        metrics as the GAN-LLM models, but WITHOUT synthetic context. Uses the real close series
        (last_input_prices), not X[:, 0], which is a PCA component and not a price."""
        from ..llm.llm_pipeline import evaluate_llm_pipeline

        X_test, y_test = shift_for_one_step_ahead(np.asarray(X_test), np.asarray(y_test))
        if last_input_prices is None:
            self.results = {"error": "LLMForecaster needs last_input_prices (the real close series)"}
            return self.results
        closes = np.asarray(last_input_prices, dtype=float).flatten()[:-1]
        cfg = {**self.config, "use_synthetic_context": False}
        query_fn = self.config.get("chat_fn")  # injection point for tests
        self.results = evaluate_llm_pipeline(closes, np.asarray(y_test, dtype=float).flatten(), X_test, cfg,
                                             seed=self.seed, generator_fn=None, mapper=None, chat_fn=query_fn)
        # Keep the column names the rest of the pipeline already reads.
        self.results["rmse"] = self.results.get("llm_rmse", float("nan"))
        self.results["mae"] = self.results.get("llm_mae", float("nan"))
        self.results["asr"] = self.results.get("llm_asr", float("nan"))
        self.results["n_llm_samples"] = self.results.get("llm_n_forecasts", 0)
        self.results["n_llm_errors"] = self.results.get("llm_n_errors", 0)
        self.results["model_id"] = self.config["model_id"]
        return self.results
