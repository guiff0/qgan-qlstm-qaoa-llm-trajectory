"""
Preprocessing: outlier detection/treatment, technical-indicator feature
engineering, and dimensionality reduction.

Outlier method is MAD (Median Absolute Deviation) throughout — matching
the manuscript's Ch.3 methodology after the IQR/MAD inconsistency there
was resolved to MAD-only. This file is the one place that decision is
implemented, so there is no risk of it drifting back to IQR in the code
the way it briefly did in the prose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def mad_outlier_mask(series: pd.Series, n_mads: float) -> pd.Series:
    """
    Returns a boolean mask of values flagged as outliers by the MAD rule:
    |x - median| > n_mads * MAD, with MAD scaled by 1.4826 so it is
    consistent with the standard deviation for normally distributed data
    (the standard scaling constant for Gaussian-consistent MAD).
    """
    median = series.median()
    mad = (series - median).abs().median() * 1.4826
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z = (series - median).abs() / mad
    return modified_z > n_mads


def domain_implausible_mask(df: pd.DataFrame, price_cols: list[str],
                             max_abs_return: float = 0.5) -> pd.Series:
    """
    Ch.3's "Cleansing Protocol" describes TWO detection passes, in this
    order: "this study first applied financial domain-based filters to
    flag physically implausible values, such as zero prices or
    minute-level returns exceeding a 50% absolute threshold. Subsequently,
    this study implemented statistical detection using [MAD]." Only the
    MAD pass existed in this file before -- this function is the first
    pass, found missing while directly checking the manuscript's data-
    cleaning claims against this file.

    Flags: any non-positive price in price_cols, OR a minute-over-minute
    return (on the first price column, conventionally 'close') whose
    absolute value exceeds max_abs_return (default 0.5 = 50%, matching
    the manuscript's stated threshold).
    """
    mask = pd.Series(False, index=df.index)
    for col in price_cols:
        mask |= df[col] <= 0
    if price_cols:
        ret = df[price_cols[0]].pct_change()
        mask |= ret.abs() > max_abs_return
    return mask


def clean_prices(df: pd.DataFrame, price_cols: list[str], n_mads_price: float = 5.0,
                  max_abs_return: float = 0.5) -> pd.DataFrame:
    """Two-pass cleaning matching Ch.3 exactly: (1) domain-based filters
    (zero/negative prices, >50% minute-over-minute returns) -- see
    domain_implausible_mask -- then (2) MAD statistical detection. Both
    passes repair with the same linear interpolation between adjacent
    valid points (matches Ch.3: 'linear interpolation between adjacent
    valid prices, preserving series continuity')."""
    df = df.copy()
    domain_mask = domain_implausible_mask(df, price_cols, max_abs_return)
    for col in price_cols:
        df.loc[domain_mask, col] = np.nan
        mad_mask = mad_outlier_mask(df[col], n_mads_price)
        df.loc[mad_mask, col] = np.nan
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
    return df


def clean_indicators(df: pd.DataFrame, indicator_cols: list[str], n_mads_indicator: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    for col in indicator_cols:
        mask = mad_outlier_mask(df[col], n_mads_indicator)
        df.loc[mask, col] = np.nan
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
    return df


def add_technical_indicators(df: pd.DataFrame, close_col: str = "close",
                              high_col: str = "high", low_col: str = "low") -> pd.DataFrame:
    """RSI, MACD, Bollinger Bands, ATR, Stochastic — the 10-feature set
    named in Ch.3 (RSI_14, MACD_line, MACD_signal, MACD_histogram,
    BB_upper, BB_lower, BB_width, ATR_14, Stoch_K_14_3, Stoch_D_14_3)."""
    df = df.copy()
    close = df[close_col]

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_line"] = ema12 - ema26
    df["MACD_signal"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
    df["MACD_histogram"] = df["MACD_line"] - df["MACD_signal"]

    # Bollinger Bands (20, 2 sigma)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # ATR(14)
    high, low = df[high_col], df[low_col]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # Stochastic %K/%D (14, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K_14_3"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["Stoch_D_14_3"] = df["Stoch_K_14_3"].rolling(3).mean()

    return df


def fit_pca(X_train: np.ndarray, n_components: int = 32) -> PCA:
    """Fit PCA on TRAINING data only, then apply the same fitted
    transform to val/test — fitting on the full dataset would leak
    test-set distribution information into the training representation."""
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)
    return pca


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
