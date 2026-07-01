"""
Synthetic transaction dataset generator for FraudGuard.

Generates ~100K realistic financial transactions with temporal ordering
and controllable fraud rate (~1.7%). Fraud patterns mimic real-world
signals: unusual hours, high amounts, foreign transactions, rapid velocity.
"""
import numpy as np
import pandas as pd

from src.config import (
    SYNTHETIC_DATASET_PATH,
    SYNTHETIC_NUM_TRANSACTIONS,
    SYNTHETIC_FRAUD_RATE,
    RANDOM_SEED,
)

MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "online_retail",
    "travel", "entertainment", "healthcare", "electronics",
    "jewelry", "cash_advance",
]

# Typical amount ranges per merchant category (mean, std)
AMOUNT_PROFILES = {
    "grocery":        (45.0,  30.0),
    "gas_station":    (40.0,  20.0),
    "restaurant":     (35.0,  25.0),
    "online_retail":  (75.0,  60.0),
    "travel":         (250.0, 200.0),
    "entertainment":  (30.0,  20.0),
    "healthcare":     (120.0, 100.0),
    "electronics":    (200.0, 180.0),
    "jewelry":        (500.0, 400.0),
    "cash_advance":   (300.0, 250.0),
}


def generate_synthetic_dataset(
    n_transactions: int = SYNTHETIC_NUM_TRANSACTIONS,
    fraud_rate: float = SYNTHETIC_FRAUD_RATE,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset with realistic fraud patterns.

    Parameters
    ----------
    n_transactions : int
        Number of transactions to generate.
    fraud_rate : float
        Fraction of transactions that are fraudulent.
    seed : int
        Random seed for reproducibility.
    save : bool
        Whether to save the CSV to disk.

    Returns
    -------
    pd.DataFrame
        Generated transaction dataset, temporally ordered.
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud

    # ── Generate legitimate transactions ─────────────────────────
    legit = _generate_legit_transactions(n_legit, rng)

    # ── Generate fraudulent transactions ─────────────────────────
    fraud = _generate_fraud_transactions(n_fraud, rng)

    # ── Combine and sort by timestamp ────────────────────────────
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = range(len(df))

    if save:
        SYNTHETIC_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SYNTHETIC_DATASET_PATH, index=False)
        print(f"✓ Saved synthetic dataset: {SYNTHETIC_DATASET_PATH}")
        print(f"  Transactions: {len(df):,}")
        print(f"  Fraud count:  {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.2%})")

    return df


def _generate_legit_transactions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate legitimate transaction records."""
    # Time spans ~90 days (in seconds)
    timestamps = np.sort(rng.uniform(0, 90 * 24 * 3600, size=n))

    categories = rng.choice(MERCHANT_CATEGORIES, size=n, p=[
        0.20, 0.12, 0.15, 0.18, 0.05, 0.10, 0.08, 0.06, 0.02, 0.04
    ])

    amounts = np.array([
        max(0.50, rng.normal(*AMOUNT_PROFILES[cat]))
        for cat in categories
    ])

    hours = (timestamps / 3600) % 24

    return pd.DataFrame({
        "timestamp": timestamps,
        "amount": np.round(amounts, 2),
        "merchant_category": categories,
        "hour_of_day": hours,
        "day_of_week": ((timestamps / 86400).astype(int) % 7),
        "is_foreign": rng.binomial(1, 0.05, size=n),  # 5% foreign
        "distance_from_home": np.abs(rng.normal(10, 15, size=n)),
        "device_trust_score": np.clip(rng.normal(0.85, 0.10, size=n), 0, 1),
        "is_fraud": 0,
    })


def _generate_fraud_transactions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate fraudulent transactions with distinguishing patterns:
    - Higher amounts (especially cash_advance, electronics, jewelry)
    - More likely at unusual hours (1am-5am)
    - Higher foreign transaction rate
    - Lower device trust scores
    - Greater distance from home
    """
    timestamps = np.sort(rng.uniform(0, 90 * 24 * 3600, size=n))

    # Fraud skews toward high-value categories
    categories = rng.choice(MERCHANT_CATEGORIES, size=n, p=[
        0.05, 0.03, 0.05, 0.20, 0.10, 0.05, 0.05, 0.15, 0.12, 0.20
    ])

    # Fraudulent amounts are typically higher
    amounts = np.array([
        max(5.0, rng.normal(AMOUNT_PROFILES[cat][0] * 3.0, AMOUNT_PROFILES[cat][1] * 2.0))
        for cat in categories
    ])

    # Fraud more likely at night (shift distribution toward 1am-5am)
    hours = rng.choice(
        np.arange(24),
        size=n,
        p=_fraud_hour_distribution(),
    ).astype(float) + rng.uniform(0, 1, size=n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "amount": np.round(amounts, 2),
        "merchant_category": categories,
        "hour_of_day": hours,
        "day_of_week": ((timestamps / 86400).astype(int) % 7),
        "is_foreign": rng.binomial(1, 0.35, size=n),  # 35% foreign (vs 5% legit)
        "distance_from_home": np.abs(rng.normal(80, 60, size=n)),  # Much farther
        "device_trust_score": np.clip(rng.normal(0.40, 0.20, size=n), 0, 1),  # Lower trust
        "is_fraud": 1,
    })


def _fraud_hour_distribution() -> np.ndarray:
    """Create hour-of-day probability distribution skewed toward nighttime."""
    probs = np.ones(24)
    # Boost late-night / early-morning hours
    probs[0:6] = 5.0    # midnight-6am: high fraud
    probs[22:24] = 3.0  # 10pm-midnight: moderate
    probs[9:17] = 0.5   # business hours: low fraud
    return probs / probs.sum()


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print(f"\nSample transactions:\n{df.head(10)}")
    print(f"\nFraud distribution:\n{df['is_fraud'].value_counts()}")
    print(f"\nMerchant category distribution (fraud only):")
    print(df[df["is_fraud"] == 1]["merchant_category"].value_counts())
