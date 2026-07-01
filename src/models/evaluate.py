"""
Model evaluation and reporting for FraudGuard.

Generates comprehensive evaluation reports including:
- PR-AUC, ROC-AUC, precision, recall, F1
- Confusion matrix
- Cost-sensitive analysis
- Threshold sweep data (for dashboard)
"""
import json
import numpy as np
import pandas as pd
import joblib

from src.config import (
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_ARTIFACT_PATH,
    EVALUATION_REPORT_PATH,
    TARGET,
)
from src.features.engineering import (
    engineer_features,
    get_feature_columns,
)
from src.utils.metrics import (
    compute_all_metrics,
    find_optimal_threshold,
    threshold_sweep,
)


def evaluate_model(
    test_df: pd.DataFrame,
    model=None,
    preprocessor=None,
    save: bool = True,
) -> dict:
    """
    Evaluate the model on the test set and produce a full report.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test split.
    model : sklearn-compatible model, optional
        If None, loads from disk.
    preprocessor : ColumnTransformer, optional
        If None, loads from disk.
    save : bool
        Whether to save the evaluation report.

    Returns
    -------
    dict
        Full evaluation report with metrics and threshold sweep.
    """
    if model is None:
        model = joblib.load(MODEL_ARTIFACT_PATH)
    if preprocessor is None:
        preprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)

    # Feature engineering
    test_df = engineer_features(test_df, fit=False)
    feature_cols = get_feature_columns(test_df)
    X_test = preprocessor.transform(test_df[feature_cols])
    y_test = test_df[TARGET].values

    # Predict probabilities
    y_scores = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    optimal_threshold, min_cost = find_optimal_threshold(y_test, y_scores)

    # Compute metrics at optimal threshold
    metrics = compute_all_metrics(y_test, y_scores, threshold=optimal_threshold)

    # Threshold sweep for dashboard
    sweep = threshold_sweep(y_test, y_scores, n_steps=100)

    report = {
        "model_type": type(model).__name__,
        "test_size": len(y_test),
        "fraud_count": int(y_test.sum()),
        "optimal_threshold": optimal_threshold,
        "min_cost": min_cost,
        "metrics_at_optimal": metrics,
        "threshold_sweep": sweep,
        "y_scores": y_scores.tolist(),
        "y_true": y_test.tolist(),
    }

    if save:
        EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Save a version without large arrays for quick loading
        report_slim = {k: v for k, v in report.items()
                       if k not in ("y_scores", "y_true", "threshold_sweep")}
        report_slim["threshold_sweep_count"] = len(sweep)

        with open(EVALUATION_REPORT_PATH, "w") as f:
            json.dump(report_slim, f, indent=2, default=_json_serializer)

        # Save full report separately (with arrays)
        full_path = EVALUATION_REPORT_PATH.parent / "evaluation_report_full.json"
        with open(full_path, "w") as f:
            json.dump(report, f, default=_json_serializer)

        print(f"✓ Evaluation report saved: {EVALUATION_REPORT_PATH}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Threshold: {optimal_threshold}")
        print(f"  Cost:      {min_cost:.0f}")

    return report


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")
