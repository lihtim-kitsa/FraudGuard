"""
Time-aware train/validation/test splitter.

Splits data chronologically (by timestamp or index order) to prevent
data leakage — critical for financial time-series data.
"""
import pandas as pd
from pathlib import Path

from src.config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    PROCESSED_DATA_DIR, TARGET,
)


def time_aware_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset chronologically (no shuffling) to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset, assumed to be in temporal order.
    train_ratio, val_ratio, test_ratio : float
        Split proportions (must sum to 1.0).
    save : bool
        Whether to persist splits to disk.

    Returns
    -------
    tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    _print_split_stats(train_df, val_df, test_df)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
        print(f"✓ Splits saved to {PROCESSED_DATA_DIR}")

    return train_df, val_df, test_df


def _print_split_stats(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Print split sizes and fraud rates."""
    total = len(train) + len(val) + len(test)
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        fraud = split[TARGET].sum()
        pct = len(split) / total * 100
        fraud_pct = fraud / len(split) * 100 if len(split) > 0 else 0
        print(f"  {name:5s}: {len(split):>7,} rows ({pct:5.1f}%) | "
              f"Fraud: {fraud:>5,} ({fraud_pct:.2f}%)")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-saved train/val/test splits from disk."""
    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    return train, val, test
