"""
Custom metrics for fraud detection.

Includes:
- PR-AUC (primary metric for imbalanced data)
- Cost-sensitive evaluation
- Optimal threshold finder
- Threshold sweep analysis
"""
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FN_COST, FP_COST


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Precision-Recall AUC (area under the PR curve)."""
    return average_precision_score(y_true, y_scores)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute a comprehensive set of metrics at a given threshold.

    Returns
    -------
    dict
        pr_auc, roc_auc, precision, recall, f1, confusion_matrix,
        total_cost, fpr, fnr
    """
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(y_true, y_scores),
        "roc_auc": roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total": total,
        "total_cost": compute_business_cost(y_true, y_pred),
    }


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> float:
    """
    Compute total business cost of the predictions.

    FN (missed fraud) is much more expensive than FP (false alarm).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn * fn_cost + fp * fp_cost


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
    n_steps: int = 100,
) -> tuple[float, float]:
    """
    Find the threshold that minimizes total business cost.

    Returns
    -------
    tuple of (optimal_threshold, min_cost)
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_threshold = 0.5
    best_cost = float("inf")

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        cost = compute_business_cost(y_true, y_pred, fn_cost, fp_cost)
        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    return round(best_threshold, 4), best_cost


def threshold_sweep(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_steps: int = 100,
) -> list[dict]:
    """
    Sweep thresholds and return metrics at each step.
    Used by the dashboard for interactive threshold visualization.
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    results = []

    for t in thresholds:
        metrics = compute_all_metrics(y_true, y_scores, threshold=t)
        results.append(metrics)

    return results
