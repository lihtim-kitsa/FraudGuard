"""
Model training orchestration for FraudGuard.

Trains multiple models (Logistic Regression baseline, XGBoost, LightGBM),
handles class imbalance via SMOTE and class weighting, and logs everything
to MLflow.
"""
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    RANDOM_SEED,
    MLFLOW_EXPERIMENT_NAME,
    MLRUNS_DIR,
    MODEL_ARTIFACT_PATH,
    TARGET,
)
from src.features.engineering import (
    engineer_features,
    build_preprocessor,
    get_feature_columns,
    transform_features,
)
from src.utils.metrics import (
    compute_all_metrics,
    compute_pr_auc,
    find_optimal_threshold,
)


def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """
    Train all model variants and return results.

    Models:
    1. Logistic Regression (baseline)
    2. XGBoost with class weighting
    3. XGBoost with SMOTE
    4. LightGBM with class weighting

    Returns
    -------
    dict
        Model name → {model, metrics, y_scores}
    """
    # ── Setup MLflow ─────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Feature engineering ──────────────────────────────────────
    print("\n🔧 Engineering features...")
    train_df = engineer_features(train_df, fit=True)
    val_df = engineer_features(val_df, fit=False)

    # ── Build preprocessor ───────────────────────────────────────
    print("🔧 Building preprocessor...")
    preprocessor = build_preprocessor(train_df, save=True)

    feature_cols = get_feature_columns(train_df)
    X_train = preprocessor.transform(train_df[feature_cols])
    y_train = train_df[TARGET].values
    X_val = preprocessor.transform(val_df[feature_cols])
    y_val = val_df[TARGET].values

    # Compute scale_pos_weight for imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"  Class balance: {n_neg:,} legit / {n_pos:,} fraud (ratio: {scale_pos_weight:.1f}:1)")

    results = {}

    # ── 1. Logistic Regression Baseline ──────────────────────────
    print("\n📊 Training: Logistic Regression (baseline)...")
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    results["logistic_regression"] = _train_and_log(
        "logistic_regression", lr_model, X_train, y_train, X_val, y_val
    )

    # ── 2. XGBoost with class weighting ──────────────────────────
    print("\n📊 Training: XGBoost (class weighted)...")
    xgb_params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}
    xgb_model = XGBClassifier(**xgb_params)
    results["xgboost_weighted"] = _train_and_log(
        "xgboost_weighted", xgb_model, X_train, y_train, X_val, y_val
    )

    # ── 3. XGBoost with SMOTE ────────────────────────────────────
    print("\n📊 Training: XGBoost (SMOTE)...")
    smote = SMOTE(random_state=RANDOM_SEED, sampling_strategy=0.3)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_resampled):,} samples "
          f"({(y_resampled == 1).sum():,} fraud)")
    xgb_smote = XGBClassifier(**XGBOOST_PARAMS)
    results["xgboost_smote"] = _train_and_log(
        "xgboost_smote", xgb_smote, X_resampled, y_resampled, X_val, y_val
    )

    # ── 4. LightGBM with class weighting ─────────────────────────
    print("\n📊 Training: LightGBM (class weighted)...")
    lgbm_params = {**LIGHTGBM_PARAMS, "scale_pos_weight": scale_pos_weight}
    lgbm_model = LGBMClassifier(**lgbm_params)
    results["lightgbm_weighted"] = _train_and_log(
        "lightgbm_weighted", lgbm_model, X_train, y_train, X_val, y_val
    )

    # ── Select best model ────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["metrics"]["pr_auc"])
    best = results[best_name]
    print(f"\n🏆 Best model: {best_name} (PR-AUC: {best['metrics']['pr_auc']:.4f})")

    # Save best model
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_ARTIFACT_PATH)
    print(f"✓ Best model saved: {MODEL_ARTIFACT_PATH}")

    return results


def _train_and_log(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train a model and log to MLflow."""
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)

        # Predict probabilities
        y_scores = model.predict_proba(X_val)[:, 1]

        # Compute metrics
        pr_auc = compute_pr_auc(y_val, y_scores)
        optimal_threshold, min_cost = find_optimal_threshold(y_val, y_scores)
        metrics = compute_all_metrics(y_val, y_scores, threshold=optimal_threshold)

        # Log to MLflow
        mlflow.log_params({
            "model_type": name,
            "optimal_threshold": optimal_threshold,
        })
        mlflow.log_metrics({
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "total_cost": metrics["total_cost"],
            "optimal_threshold": optimal_threshold,
        })
        # Log model artifact (best-effort — some model types require extra trust config)
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            pass  # Model saved via joblib separately

        print(f"  PR-AUC: {pr_auc:.4f} | F1: {metrics['f1']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
              f"Threshold: {optimal_threshold}")

    return {
        "model": model,
        "metrics": metrics,
        "y_scores": y_scores,
        "optimal_threshold": optimal_threshold,
    }
