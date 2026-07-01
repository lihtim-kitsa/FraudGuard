"""
Data loader — loads either the Kaggle Credit Card Fraud CSV or the
synthetic dataset, validates schema, and returns a clean DataFrame.
"""
import pandas as pd
from pathlib import Path

from src.config import (
    KAGGLE_DATASET_PATH,
    SYNTHETIC_DATASET_PATH,
    TARGET,
)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Load a transaction dataset from disk.

    Priority:
    1. Explicit path (if provided)
    2. Kaggle CSV (if exists)
    3. Synthetic CSV (generate if missing)

    Parameters
    ----------
    path : Path, optional
        Explicit path to a CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded and validated dataset.
    """
    if path is not None:
        csv_path = Path(path)
    elif KAGGLE_DATASET_PATH.exists():
        csv_path = KAGGLE_DATASET_PATH
        print(f"📂 Loading Kaggle dataset: {csv_path}")
    elif SYNTHETIC_DATASET_PATH.exists():
        csv_path = SYNTHETIC_DATASET_PATH
        print(f"📂 Loading synthetic dataset: {csv_path}")
    else:
        print("⚡ No dataset found — generating synthetic data...")
        from src.data.generate_synthetic import generate_synthetic_dataset
        df = generate_synthetic_dataset(save=True)
        return _validate(df)

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    return _validate(df)


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic validation: ensure target exists, no all-null columns."""
    if TARGET not in df.columns:
        # Handle Kaggle dataset which uses 'Class' instead of 'is_fraud'
        if "Class" in df.columns:
            df = df.rename(columns={"Class": TARGET})
        else:
            raise ValueError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")

    # Drop fully null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        print(f"  ⚠ Dropping all-null columns: {null_cols}")
        df = df.drop(columns=null_cols)

    # Report basic stats
    fraud_count = df[TARGET].sum()
    print(f"  Fraud: {fraud_count:,} / {len(df):,} ({fraud_count/len(df):.2%})")

    return df
