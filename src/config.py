"""
Central configuration for FraudGuard.
All paths, hyperparameters, and feature definitions in one place.
"""
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, MLRUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
SYNTHETIC_DATASET_PATH = RAW_DATA_DIR / "synthetic_transactions.csv"
KAGGLE_DATASET_PATH = RAW_DATA_DIR / "creditcard.csv"
SYNTHETIC_NUM_TRANSACTIONS = 100_000
SYNTHETIC_FRAUD_RATE = 0.017  # ~1.7%
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "merchant_category",
    "day_of_week",
]

NUMERICAL_FEATURES = [
    "amount",
    "hour_of_day",
    "is_foreign",
    "distance_from_home",
    "time_since_last_txn",
    "velocity_1h",
    "velocity_24h",
    "amount_zscore",
    "device_trust_score",
    "hour_sin",
    "hour_cos",
    "amount_log",
]

TARGET = "is_fraud"

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# ──────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "fraudguard"

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "aucpr",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# ──────────────────────────────────────────────
# Business logic
# ──────────────────────────────────────────────
# Cost of a false negative (missed fraud) vs false positive (blocked legit)
# A missed fraud is typically 10-50x more costly than a false alarm
FN_COST = 50.0   # Cost multiplier for missing a fraud
FP_COST = 1.0    # Cost multiplier for false alarm

# Decision thresholds (default)
DEFAULT_DECLINE_THRESHOLD = 0.7    # Above this → auto-decline
DEFAULT_REVIEW_THRESHOLD = 0.3     # Between review & decline → flag for review
# Below review threshold → auto-approve

# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
DB_PATH = PROJECT_ROOT / "fraudguard.db"
MODEL_ARTIFACT_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_ARTIFACT_PATH = MODELS_DIR / "preprocessor.joblib"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.json"
