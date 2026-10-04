"""
Prepares processed train/val/test .npy arrays from the raw data files
you place under data/raw/ (see SETUP.md).

Run with:  python -m scripts.prepare_data
"""
from __future__ import annotations

import os
import sys
import time
import psutil
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import (
    load_dukascopy, load_fred, load_vix, merge_all_sources,
    temporal_split, save_processed_arrays,
)
from src.data.preprocessing import (
    clean_prices, clean_indicators, add_technical_indicators, fit_pca, fit_scaler,
)
from src.utils.config import load_config
from src.utils.progress import progress_bar


INDICATOR_COLS = [
    "RSI_14", "MACD_line", "MACD_signal", "MACD_histogram",
    "BB_upper", "BB_lower", "BB_width", "ATR_14", "Stoch_K_14_3", "Stoch_D_14_3",
]
PRICE_COLS = ["open", "high", "low", "close"]


def expand_feature_space(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Expands the technical and price feature set to ensure total raw features >= 32
    for high-dimensional PCA projection without producing all-NaN series.
    """
    df = df.copy()

    # Multi-period Returns & Volatility (Note: rolling std requires period >= 2)
    for p in [1, 3, 5, 10, 15, 30]:
        df[f"return_{p}"] = df["close"].pct_change(p)
        df[f"log_return_{p}"] = np.log(df["close"] / df["close"].shift(p))
        if p >= 2:
            df[f"volatility_{p}"] = df[f"return_1"].rolling(p).std()

    # Moving Averages & Relative Ratios
    for ma in [5, 10, 20, 50]:
        df[f"sma_{ma}"] = df["close"].rolling(ma).mean()
        df[f"ema_{ma}"] = df["close"].ewm(span=ma, adjust=False).mean()
        df[f"close_to_sma_{ma}"] = df["close"] / (df[f"sma_{ma}"] + 1e-8)

    # Momentum & Range Indicators
    df["high_low_ratio"] = df["high"] / (df["low"] + 1e-8)
    df["close_open_ratio"] = df["close"] / (df["open"] + 1e-8)
    df["volume_change"] = df["volume"].pct_change(1)
    df["volume_sma_10"] = df["volume"].rolling(10).mean()

    # Replace potential infinite values resulting from zero-division
    df = df.replace([np.inf, -np.inf], np.nan)

    # Extract all feature columns excluding timestamp and date columns
    feature_cols = [c for c in df.columns if c not in ["timestamp", "date"]]

    return df, feature_cols


_PIPELINE_BAR = None  # set inside main(); module-level so log_step() can update it


def log_step(step_num: int, total_steps: int, title: str):
    """Prints a clear section divider AND advances the overall pipeline
    progress bar, so -- even though each of the 7 phases below is mostly
    one or two vectorized pandas/sklearn calls rather than a Python loop
    a per-row bar could attach to -- there is still a live, persistent
    percentage on screen for the whole run (e.g. "3/7 phases, 43%")."""
    divider = "=" * 70
    print(f"\n{divider}")
    print(f"[{step_num}/{total_steps}] {title.upper()}")
    print(f"{divider}")
    if _PIPELINE_BAR is not None:
        _PIPELINE_BAR.n = step_num - 1
        _PIPELINE_BAR.set_postfix(phase=title)
        _PIPELINE_BAR.refresh() if hasattr(_PIPELINE_BAR, "refresh") else None


def get_mem_mb() -> float:
    """Returns the current process memory consumption in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def leakage_check(X_train, y_train, X_val, y_val, max_rows: int = 200_000) -> bool:
    """Diagnostic on the RAW, same-row (X_t, y_t) pairing saved to
    X_train.npy/y_train.npy: fit a plain linear map X_t -> y_t and score
    it on validation. This WILL still flag leakage here, by design --
    y stays same-row in these saved arrays because Classical LSTM's
    WindowedSequenceDataset needs that exact convention (see the comment
    above this function's call site in main()). The actual fix for the
    three same-row models (Classical GAN-LLM, QGAN-LLM, QLSTM Forecaster)
    is applied locally by each of them via
    src/evaluation/one_step_ahead.py's shift_for_one_step_ahead(), not
    here -- this function remains a trip-wire on the shared raw arrays,
    not a statement about what those three models actually train on.

    One-step-ahead forecasting of a price series cannot beat 'persistence'
    (predict the next value = the current value) by much; if a *linear* map of
    the same row's features gets a validation RMSE BELOW the RMS one-step change
    of y itself, y_t is recoverable from X_t rather than needing to be forecast.
    Returns True if that red flag is raised."""
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(X_train), size=min(max_rows, len(X_train)), replace=False))
    A = np.c_[X_train[idx], np.ones(len(idx))]
    w = np.linalg.lstsq(A, y_train[idx].astype(np.float64), rcond=None)[0]
    pred = np.c_[X_val, np.ones(len(X_val))] @ w
    same_row_rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
    persistence_rmse = float(np.sqrt(np.mean(np.diff(y_val.astype(np.float64)) ** 2)))
    print(f" -> Same-row linear map X_t -> y_t : val RMSE = {same_row_rmse:.3e}")
    print(f" -> One-step persistence floor      : val RMSE = {persistence_rmse:.3e}")
    if same_row_rmse < persistence_rmse:
        print("\n" + "!" * 70)
        print("!! TARGET LEAKAGE: y_t is recoverable from X_t better than the best possible")
        print("!! one-step-ahead forecast. Models trained on row-aligned (X_t, y_t) pairs are")
        print("!! NOT forecasting. `close` is both an input feature and the target.")
        print("!! Their RMSE is not comparable to the windowed Classical LSTM's next-step RMSE.")
        print("!" * 70 + "\n")
        return True
    return False


def main():
    global _PIPELINE_BAR
    total_start_time = time.time()
    total_steps = 7
    _PIPELINE_BAR = progress_bar(total=total_steps, desc="prepare_data.py overall", unit="phase")

    log_step(1, total_steps, "Loading Configurations")
    cfg = load_config()
    data_cfg = cfg["data"]
    target_pca_components = cfg["model"]["n_features"]

    print(f" -> Processed Output Directory: {data_cfg['processed_dir']}")
    print(f" -> Target PCA Features: {target_pca_components}")
    print(f" -> Initial RAM usage: {get_mem_mb():.2f} MB")

    log_step(2, total_steps, "Loading Raw Data Sources")
    t0 = time.time()

    print(" -> Loading Dukascopy FX dataset...")
    dukascopy_df = load_dukascopy(data_cfg["dukascopy_file"])
    print(f"    [Dukascopy] {len(dukascopy_df):,} rows | Span: {dukascopy_df['timestamp'].min()} -> {dukascopy_df['timestamp'].max()}")

    print(" -> Loading FRED Macroeconomic dataset...")
    fred_df = load_fred(data_cfg["fred_file"])
    date_col = 'date' if 'date' in fred_df.columns else fred_df.index.name
    min_date = fred_df[date_col].min() if date_col in fred_df.columns else fred_df.index.min()
    max_date = fred_df[date_col].max() if date_col in fred_df.columns else fred_df.index.max()
    print(f"    [FRED] {len(fred_df):,} rows | Span: {min_date} -> {max_date}")

    print(" -> Loading VIX volatility index...")
    # Start a couple of weeks BEFORE the study window: the merge applies each
    # daily VIX close only from the next day, so the first bars need a prior obs.
    vix_start = (pd.Timestamp(data_cfg["train_start"]) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    vix_df = load_vix(data_cfg["yfinance_ticker"], start=vix_start, end=data_cfg["test_end"])
    print(f"    [VIX] {len(vix_df):,} rows | {vix_df['date'].min().date()} -> {vix_df['date'].max().date()}")

    print(f" -> Step completed in {time.time() - t0:.2f}s | RAM: {get_mem_mb():.2f} MB")

    log_step(3, total_steps, "Merging Sources & Alignment")
    t0 = time.time()
    print(" -> Aligning daily/monthly Macro+VIX series onto 1-minute bars (publication-lag aware, no look-ahead)...")
    merged = merge_all_sources(
        dukascopy_df, fred_df, vix_df,
        release_lags=data_cfg.get("fred_release_lags"),       # optional {SERIES: days} override
        vix_lag_days=data_cfg.get("vix_lag_days", 1),
    )
    print(f" -> Consolidated Master DataFrame: {len(merged):,} rows | Columns: {list(merged.columns)}")
    print(f" -> Step completed in {time.time() - t0:.2f}s | RAM: {get_mem_mb():.2f} MB")

    log_step(4, total_steps, "Cleaning Outliers & Technical Indicator Engineering")
    t0 = time.time()
    print(f" -> Filtering price outliers (MAD threshold: {data_cfg['outlier_threshold_price']})...")
    merged = clean_prices(merged, PRICE_COLS, n_mads_price=data_cfg["outlier_threshold_price"],
                           max_abs_return=data_cfg.get("max_abs_return", 0.5))

    print(" -> Calculating primary technical indicators...")
    merged = add_technical_indicators(merged)

    print(f" -> Filtering primary indicator outliers (MAD threshold: {data_cfg['outlier_threshold_indicator']})...")
    indicator_cols_to_clean = [c for c in INDICATOR_COLS if c in merged.columns]
    merged = clean_indicators(merged, indicator_cols_to_clean, n_mads_indicator=data_cfg["outlier_threshold_indicator"])

    print(" -> Expanding feature space to satisfy n_features=32 PCA requirement...")
    merged, feature_cols = expand_feature_space(merged)

    print(" -> Dropping NaN warmup window rows...")
    initial_rows = len(merged)
    merged = merged.dropna().reset_index(drop=True)
    print(f" -> Final clean dataset size: {len(merged):,} rows (dropped {initial_rows - len(merged):,} warmup/outlier rows)")
    print(f" -> Step completed in {time.time() - t0:.2f}s | RAM: {get_mem_mb():.2f} MB")

    log_step(5, total_steps, "Chronological Temporal Splitting")
    t0 = time.time()
    splits = temporal_split(merged, data_cfg)
    for name, split_df in splits.items():
        if len(split_df) > 0:
            print(f" -> Split '{name:<5}': {len(split_df):>10,} rows | {split_df['timestamp'].min()} to {split_df['timestamp'].max()}")
        else:
            print(f" [!] Split '{name:<5}': 0 rows (Verify date bounds in config/default_config.yaml)")
    print(f" -> Step completed in {time.time() - t0:.2f}s")

    log_step(6, total_steps, "Scaling & Dimensionality Reduction (PCA)")
    t0 = time.time()

    target_col = "close"
    n_raw_features = len(feature_cols)
    print(f" -> Total raw features extracted: {n_raw_features}")
    print(f" -> Raw Feature Set: {feature_cols}")

    if n_raw_features < target_pca_components:
        raise ValueError(
            f"Cannot fit PCA with n_components={target_pca_components}. "
            f"Only {n_raw_features} raw features available."
        )

    print(" -> Fitting StandardScaler on TRAINING split only...")
    scaler = fit_scaler(splits["train"][feature_cols].to_numpy())

    print(" -> Transforming training set using StandardScaler...")
    train_scaled = scaler.transform(splits["train"][feature_cols].to_numpy())

    print(f" -> Fitting PCA (n_components={target_pca_components}) on scaled training set...")
    pca = fit_pca(train_scaled, n_components=target_pca_components)

    var_ratios = pca.explained_variance_ratio_
    cum_var = np.sum(var_ratios)
    print(f" -> [TRACE] PCA fitted successfully with n_components={target_pca_components}.")
    print(f" -> PCA Explained Variance Ratios: {np.round(var_ratios, 4)}")
    print(f" -> Cumulative Variance Retained: {cum_var:.4%}")
    print(f" -> Step completed in {time.time() - t0:.2f}s | RAM: {get_mem_mb():.2f} MB")

    log_step(7, total_steps, "Array Export & Verification")
    t0 = time.time()
    os.makedirs(data_cfg["processed_dir"], exist_ok=True)

    for split_name, split_df in progress_bar(list(splits.items()), total=len(splits),
                                             desc="Exporting splits", unit="split"):
        print(f" -> Processing '{split_name}' array transformations...")

        raw_matrix = split_df[feature_cols].to_numpy()
        X_scaled = scaler.transform(raw_matrix)
        X_pca = pca.transform(X_scaled).astype(np.float32)
        y = split_df[target_col].to_numpy(dtype=np.float32)
        # y[k] stays SAME-ROW (close[k]) here deliberately -- do not "fix"
        # the leakage by shifting this shared array. Classical LSTM's
        # WindowedSequenceDataset already correctly relies on exactly this
        # convention (target = y[idx+sequence_length], predicted from a
        # window [idx, idx+sequence_length) that never includes that row),
        # so shifting y here would silently introduce an off-by-one error
        # into the one model that was already correct. The actual fix for
        # the same-row models (Classical GAN-LLM, QGAN-LLM, QLSTM Forecaster)
        # is in their own training/eval code -- see
        # src/evaluation/one_step_ahead.py's shift_for_one_step_ahead(),
        # applied locally by each of those three models where they build
        # their (X, y) pairs, not here.

        x_path = os.path.join(data_cfg["processed_dir"], f"X_{split_name}.npy")
        y_path = os.path.join(data_cfg["processed_dir"], f"y_{split_name}.npy")

        print(f" -> [TRACE] Writing arrays to disk for split '{split_name}'...")
        np.save(x_path, X_pca)
        np.save(y_path, y)

        x_size_mb = os.path.getsize(x_path) / (1024 * 1024)
        y_size_mb = os.path.getsize(y_path) / (1024 * 1024)

        print(f"    [EXPORTED] X_{split_name}.npy -> Shape: {X_pca.shape} | Size: {x_size_mb:.2f} MB")
        print(f"    [EXPORTED] y_{split_name}.npy -> Shape: {y.shape}    | Size: {y_size_mb:.2f} MB")

    print(" -> Leakage check on exported arrays (expected to still flag the "
          "RAW same-row pairing -- see leakage_check's docstring; the three "
          "affected models fix this themselves via shift_for_one_step_ahead())...")
    Xt = np.load(os.path.join(data_cfg["processed_dir"], "X_train.npy"), mmap_mode="r")
    yt = np.load(os.path.join(data_cfg["processed_dir"], "y_train.npy"))
    Xv = np.load(os.path.join(data_cfg["processed_dir"], "X_val.npy"))
    yv = np.load(os.path.join(data_cfg["processed_dir"], "y_val.npy"))
    leakage_check(np.asarray(Xt), yt, Xv, yv)

    _PIPELINE_BAR.n = total_steps
    _PIPELINE_BAR.close()

    total_time = time.time() - total_start_time
    print("\n" + "=" * 70)
    print(f"SUCCESS: Data preparation completed in {total_time:.2f} seconds.")
    print(f"Processed arrays saved to -> {data_cfg['processed_dir']}/")
    print("Next step: python -m src.experiments.run_all")
    print("=" * 70)


if __name__ == "__main__":
    main()
