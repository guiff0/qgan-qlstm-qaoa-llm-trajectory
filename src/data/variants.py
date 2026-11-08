"""Preprocessing variants for the sensitivity analysis (reviewer claim 6).

The baseline pipeline (scripts/prepare_data.py without --variant) is untouched. A variant changes ONE thing and writes to
its own folder (data/processed_variants/<variant>/), with a variant.json that records what changed and by how much.

  baseline              as published (volume untreated, indicators repaired by interpolation, StandardScaler)
  volume_winsor         raw volume clipped at the TRAINING-period 99.9th percentile before volume features are built
                        (a corrupt 446-million-unit bar no longer drives the scaler and PCA)
  robust_scaler         RobustScaler (median / IQR, fitted on training rows) replaces StandardScaler for every feature
  no_interpolation      indicator outliers are NOT flagged or repaired (the 15.3% interpolation step is skipped)
  documented_features   adds the rolling skewness and kurtosis features the manuscript describes (5/10/21/63-bar windows
                        of the one-bar return); interaction terms are not generated

Checkpoints are keyed by config + seed only, not by data. A variant therefore needs its own config (see
scripts/run_sensitivity_study.py), which adds `data_variant` to the GAN sections so a baseline checkpoint can never be
resumed on different data.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

VARIANTS = ("baseline", "volume_winsor", "robust_scaler", "no_interpolation", "documented_features")
WINSOR_QUANTILE = 0.999
DOC_WINDOWS = (5, 10, 21, 63)


def check_variant(name: str) -> str:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose from {VARIANTS}")
    return name


def default_processed_dir(variant: str, root: str = os.path.join("data", "processed_variants")) -> str:
    return os.path.join(root, check_variant(variant))


def skips_indicator_repair(variant: str) -> bool:
    return variant == "no_interpolation"


def pre_features(df: pd.DataFrame, variant: str, data_cfg: dict, q: float = WINSOR_QUANTILE):
    """Applied to the merged frame BEFORE feature expansion. Returns (frame, info)."""
    info: dict = {}
    if variant == "volume_winsor":
        train = df["timestamp"] <= pd.Timestamp(data_cfg["train_end"]) if "train_end" in data_cfg else slice(None)
        cap = float(df.loc[train, "volume"].quantile(q))
        before_max = float(df["volume"].max())
        n_clipped = int((df["volume"] > cap).sum())
        df["volume"] = df["volume"].clip(upper=cap)
        info = {"volume_cap": cap, "quantile": q, "fitted_on": "training period only", "max_volume_before": before_max,
                "n_clipped": n_clipped, "share_clipped": n_clipped / max(len(df), 1)}
    return df, info


def post_features(df: pd.DataFrame, feature_cols: list, variant: str):
    """Applied after expand_feature_space, before NaN rows are dropped. Returns (frame, feature_cols)."""
    if variant == "documented_features":
        ret = df["close"].pct_change(1)
        new = []
        for w in DOC_WINDOWS:
            df[f"roll_skew_{w}"] = ret.rolling(w).skew()
            df[f"roll_kurt_{w}"] = ret.rolling(w).kurt()
            new += [f"roll_skew_{w}", f"roll_kurt_{w}"]
        df = df.replace([np.inf, -np.inf], np.nan)
        feature_cols = list(feature_cols) + new
    return df, feature_cols


def fit_variant_scaler(variant: str, X_train: np.ndarray):
    """Scaler with transform / inverse_transform (the saved scaler maps synthetic rows back to prices)."""
    if variant == "robust_scaler":
        from sklearn.preprocessing import RobustScaler
        return RobustScaler().fit(X_train)
    from src.data.preprocessing import fit_scaler
    return fit_scaler(X_train)


def write_manifest(processed_dir: str, variant: str, info: dict, feature_cols: list, n_rows: dict) -> str:
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, "variant.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "details": info, "n_features_raw": len(feature_cols),
                   "feature_cols": list(feature_cols), "n_rows": n_rows}, f, indent=2, default=str)
    return path
