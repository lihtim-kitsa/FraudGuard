"""
Threshold routes — GET/PUT /threshold, GET /threshold/sweep

Allows business users to adjust decision thresholds and see
the precision/recall tradeoff in real time.
"""
from fastapi import APIRouter, HTTPException

from api.schemas import ThresholdConfig, ThresholdResponse, ThresholdSweepPoint
from api.services.model_service import model_service
from api.services.monitor_service import monitor_service

router = APIRouter(prefix="/threshold", tags=["Threshold"])


@router.get("", response_model=ThresholdResponse)
async def get_threshold():
    """Get current threshold configuration and resulting metrics."""
    metrics = monitor_service.get_threshold_metrics(model_service.decline_threshold)

    if not metrics:
        # Return defaults if no evaluation data
        return ThresholdResponse(
            decline_threshold=model_service.decline_threshold,
            review_threshold=model_service.review_threshold,
            precision=0, recall=0, f1=0, pr_auc=0,
            total_cost=0, tp=0, fp=0, tn=0, fn=0,
        )

    return ThresholdResponse(
        decline_threshold=model_service.decline_threshold,
        review_threshold=model_service.review_threshold,
        precision=metrics.get("precision", 0),
        recall=metrics.get("recall", 0),
        f1=metrics.get("f1", 0),
        pr_auc=metrics.get("pr_auc", 0),
        total_cost=metrics.get("total_cost", 0),
        tp=metrics.get("tp", 0),
        fp=metrics.get("fp", 0),
        tn=metrics.get("tn", 0),
        fn=metrics.get("fn", 0),
    )


@router.put("", response_model=ThresholdResponse)
async def update_threshold(config: ThresholdConfig):
    """
    Update the decision thresholds.

    Returns updated metrics at the new threshold.
    """
    if config.review_threshold >= config.decline_threshold:
        raise HTTPException(
            status_code=400,
            detail="review_threshold must be less than decline_threshold"
        )

    model_service.update_thresholds(config.decline_threshold, config.review_threshold)
    metrics = monitor_service.get_threshold_metrics(config.decline_threshold)

    if not metrics:
        return ThresholdResponse(
            decline_threshold=config.decline_threshold,
            review_threshold=config.review_threshold,
            precision=0, recall=0, f1=0, pr_auc=0,
            total_cost=0, tp=0, fp=0, tn=0, fn=0,
        )

    return ThresholdResponse(
        decline_threshold=config.decline_threshold,
        review_threshold=config.review_threshold,
        precision=metrics.get("precision", 0),
        recall=metrics.get("recall", 0),
        f1=metrics.get("f1", 0),
        pr_auc=metrics.get("pr_auc", 0),
        total_cost=metrics.get("total_cost", 0),
        tp=metrics.get("tp", 0),
        fp=metrics.get("fp", 0),
        tn=metrics.get("tn", 0),
        fn=metrics.get("fn", 0),
    )


@router.get("/sweep", response_model=list[ThresholdSweepPoint])
async def threshold_sweep():
    """
    Return metrics at every 0.01 threshold increment.

    Used by the dashboard to render the precision/recall/cost tradeoff curve.
    """
    sweep = monitor_service.get_threshold_sweep()

    if not sweep:
        raise HTTPException(
            status_code=404,
            detail="No threshold sweep data available. Run the training pipeline first."
        )

    return [
        ThresholdSweepPoint(
            threshold=pt.get("threshold", 0),
            precision=pt.get("precision", 0),
            recall=pt.get("recall", 0),
            f1=pt.get("f1", 0),
            total_cost=pt.get("total_cost", 0),
            tp=pt.get("tp", 0),
            fp=pt.get("fp", 0),
            tn=pt.get("tn", 0),
            fn=pt.get("fn", 0),
        )
        for pt in sweep
    ]
