"""
SHAP-based model explainability for FraudGuard.

Provides:
- Global feature importance (mean |SHAP values|)
- Per-transaction explanations (top contributing features)
- Human-readable explanation strings
"""
import numpy as np
import pandas as pd
import shap
import joblib

from src.config import (
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_ARTIFACT_PATH,
    SHAP_EXPLAINER_PATH,
    MODELS_DIR,
)
from src.features.engineering import get_feature_columns, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def create_shap_explainer(
    model=None,
    preprocessor=None,
    X_background: np.ndarray | None = None,
    save: bool = True,
) -> shap.TreeExplainer:
    """
    Create and optionally save a SHAP TreeExplainer.

    Parameters
    ----------
    model : trained model
        If None, loads from disk.
    preprocessor : ColumnTransformer
        If None, loads from disk. Used to get feature names.
    X_background : np.ndarray, optional
        Background dataset for the explainer (subset of training data).
    save : bool
        Whether to save the explainer.

    Returns
    -------
    shap.TreeExplainer
    """
    if model is None:
        model = joblib.load(MODEL_ARTIFACT_PATH)

    explainer = shap.TreeExplainer(model)

    if save:
        SHAP_EXPLAINER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(explainer, SHAP_EXPLAINER_PATH)
        print(f"✓ SHAP explainer saved: {SHAP_EXPLAINER_PATH}")

    return explainer


def get_feature_names(preprocessor=None) -> list[str]:
    """Get human-readable feature names from the preprocessor."""
    if preprocessor is None:
        preprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)

    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)

    return feature_names


def explain_single(
    features: np.ndarray,
    explainer: shap.TreeExplainer | None = None,
    feature_names: list[str] | None = None,
    top_k: int = 5,
) -> dict:
    """
    Explain a single prediction using SHAP values.

    Parameters
    ----------
    features : np.ndarray
        1D or 2D array of preprocessed features for one transaction.
    explainer : shap.TreeExplainer
        Pre-loaded explainer.
    feature_names : list[str]
        Human-readable feature names.
    top_k : int
        Number of top features to include.

    Returns
    -------
    dict
        {
            "top_features": [{"feature": str, "shap_value": float, "direction": str}, ...],
            "base_value": float,
            "explanation_text": str
        }
    """
    if explainer is None:
        explainer = joblib.load(SHAP_EXPLAINER_PATH)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    shap_values = explainer.shap_values(features)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Binary classification: use class 1 (fraud) SHAP values
        sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        sv = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    # Get feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(sv))]

    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(sv))[::-1][:top_k]

    top_features = []
    for idx in indices:
        val = float(sv[idx])
        top_features.append({
            "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
            "shap_value": round(val, 4),
            "direction": "increases fraud risk" if val > 0 else "decreases fraud risk",
        })

    # Build human-readable explanation
    explanation_parts = []
    for feat in top_features[:3]:
        sign = "↑" if feat["shap_value"] > 0 else "↓"
        explanation_parts.append(
            f"{sign} {feat['feature']}: {feat['shap_value']:+.3f}"
        )
    explanation_text = " | ".join(explanation_parts)

    return {
        "top_features": top_features,
        "base_value": float(base_value),
        "explanation_text": explanation_text,
    }


def global_feature_importance(
    X: np.ndarray,
    explainer: shap.TreeExplainer | None = None,
    feature_names: list[str] | None = None,
    max_samples: int = 1000,
) -> list[dict]:
    """
    Compute global feature importance using mean |SHAP values|.

    Returns a sorted list of {feature, importance}.
    """
    if explainer is None:
        explainer = joblib.load(SHAP_EXPLAINER_PATH)

    # Subsample for performance
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

    importance = []
    for i, val in enumerate(mean_abs):
        importance.append({
            "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
            "importance": round(float(val), 4),
        })

    importance.sort(key=lambda x: x["importance"], reverse=True)
    return importance
