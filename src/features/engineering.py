"""
Feature engineering pipeline for FraudGuard.

Transforms raw transaction data into model-ready features:
- Transaction velocity (1h, 24h windows)
- Time since last transaction
- Amount z-scores (per merchant category)
- Cyclical hour encoding (sin/cos)
- Log-transformed amounts
- One-hot encoding for categoricals
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    PREPROCESSOR_ARTIFACT_PATH,
)


def engineer_features(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Apply feature engineering to a transaction DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data.
    fit : bool
        If True, this is the training set — compute rolling stats from scratch.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features added.
    """
    df = df.copy()

    # ── Velocity features ────────────────────────────────────────
    if "timestamp" in df.columns:
        df["velocity_1h"] = _rolling_count(df["timestamp"], window_seconds=3600)
        df["velocity_24h"] = _rolling_count(df["timestamp"], window_seconds=86400)
        df["time_since_last_txn"] = df["timestamp"].diff().fillna(0).clip(lower=0)
    else:
        # Fallback: use index as proxy for time ordering
        for col in ["velocity_1h", "velocity_24h", "time_since_last_txn"]:
            if col not in df.columns:
                df[col] = 0.0

    # ── Amount z-score per merchant category ─────────────────────
    if "merchant_category" in df.columns and "amount" in df.columns:
        df["amount_zscore"] = df.groupby("merchant_category")["amount"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    elif "amount_zscore" not in df.columns:
        df["amount_zscore"] = 0.0

    # ── Cyclical hour encoding ───────────────────────────────────
    if "hour_of_day" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 0.0

    # ── Log-transform amount ─────────────────────────────────────
    if "amount" in df.columns:
        df["amount_log"] = np.log1p(df["amount"])
    elif "amount_log" not in df.columns:
        df["amount_log"] = 0.0

    # Fill any remaining NaNs
    df = df.fillna(0)

    return df


def _rolling_count(
    timestamps: pd.Series, window_seconds: int
) -> pd.Series:
    """
    Count how many transactions occurred in the preceding `window_seconds`.
    Uses a vectorized approach for performance.
    """
    ts = timestamps.values.astype(float)
    counts = np.zeros(len(ts))
    for i in range(1, len(ts)):
        window_start = ts[i] - window_seconds
        # Count transactions in [window_start, ts[i])
        counts[i] = np.searchsorted(ts[:i], ts[i]) - np.searchsorted(ts[:i], window_start)
    return pd.Series(counts, index=timestamps.index)


def build_preprocessor(
    df: pd.DataFrame, save: bool = True
) -> ColumnTransformer:
    """
    Build and fit a sklearn ColumnTransformer for the feature set.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (already feature-engineered).
    save : bool
        Whether to persist the fitted preprocessor.

    Returns
    -------
    ColumnTransformer
        Fitted preprocessor.
    """
    # Determine which configured features actually exist in the data
    num_feats = [f for f in NUMERICAL_FEATURES if f in df.columns]
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in df.columns]

    transformers = []

    if num_feats:
        transformers.append(("num", StandardScaler(), num_feats))

    if cat_feats:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_feats,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    feature_cols = num_feats + cat_feats
    preprocessor.fit(df[feature_cols])

    if save:
        PREPROCESSOR_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_ARTIFACT_PATH)
        print(f"✓ Preprocessor saved: {PREPROCESSOR_ARTIFACT_PATH}")

    return preprocessor


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns present in the DataFrame."""
    num_feats = [f for f in NUMERICAL_FEATURES if f in df.columns]
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    return num_feats + cat_feats


def transform_features(
    df: pd.DataFrame, preprocessor: ColumnTransformer
) -> np.ndarray:
    """Transform features using a fitted preprocessor."""
    feature_cols = get_feature_columns(df)
    return preprocessor.transform(df[feature_cols])
