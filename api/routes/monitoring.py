"""
Monitoring routes — GET /metrics, /drift, /predictions, /feature-importance

Provides model performance, drift data, and prediction history
for the monitoring dashboard.
"""
from fastapi import APIRouter

from api.schemas import (
    ModelMetrics,
    FeatureImportanceItem,
    DriftPoint,
    PredictionLog,
)
from api.services.monitor_service import monitor_service
from api.database import get_recent_predictions, get_prediction_count, get_prediction_stats

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/metrics", response_model=ModelMetrics)
async def get_metrics():
    """Get current model performance metrics."""
    metrics = monitor_service.get_metrics()
    stats = get_prediction_stats()

    return ModelMetrics(
        model_type=metrics.get("model_type", "unknown"),
        pr_auc=metrics.get("pr_auc", 0),
        roc_auc=metrics.get("roc_auc", 0),
        precision=metrics.get("precision", 0),
        recall=metrics.get("recall", 0),
        f1=metrics.get("f1", 0),
        threshold=metrics.get("threshold", 0.5),
        total_predictions=stats.get("total", 0),
        fraud_rate=metrics.get("fraud_rate", 0),
    )


@router.get("/feature-importance", response_model=list[FeatureImportanceItem])
async def get_feature_importance():
    """Get global SHAP feature importance ranking."""
    importance = monitor_service.get_feature_importance()
    return [
        FeatureImportanceItem(feature=item["feature"], importance=item["importance"])
        for item in importance
    ]


@router.get("/drift", response_model=list[DriftPoint])
async def get_drift():
    """
    Get feature drift data over time windows.

    In production, this compares live feature distributions against
    the training data baseline. Here we provide simulated drift data
    for demonstration purposes.
    """
    drift_data = monitor_service.get_drift_data()
    return [
        DriftPoint(
            window=item["window"],
            feature=item["feature"],
            mean_shift=item["mean_shift"],
            std_shift=item["std_shift"],
            drift_score=item["drift_score"],
        )
        for item in drift_data
    ]


@router.get("/predictions/recent", response_model=list[PredictionLog])
async def get_recent():
    """Get the most recent scored transactions from the prediction log."""
    predictions = get_recent_predictions(limit=50)
    return [
        PredictionLog(
            id=p["id"],
            timestamp=p["timestamp"],
            amount=p.get("amount", 0),
            fraud_probability=p["fraud_probability"],
            decision=p["decision"],
            latency_ms=p.get("latency_ms", 0),
        )
        for p in predictions
    ]


@router.get("/predictions/stats")
async def get_stats():
    """Get aggregate prediction statistics."""
    return get_prediction_stats()
