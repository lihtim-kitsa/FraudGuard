"""
Scoring routes — POST /score and POST /score/batch

Scores transactions for fraud probability and returns decisions
with SHAP explanations.
"""
import uuid
from fastapi import APIRouter, HTTPException

from api.schemas import (
    TransactionRequest,
    ScoringResponse,
    FeatureExplanation,
    BatchScoringRequest,
    BatchScoringResponse,
)
from api.services.model_service import model_service
from api.services.explainer_service import explainer_service
from api.database import log_prediction

router = APIRouter(prefix="/score", tags=["Scoring"])


@router.post("", response_model=ScoringResponse)
async def score_transaction(txn: TransactionRequest):
    """
    Score a single transaction for fraud risk.

    Returns fraud probability, decision (approve/review/decline),
    risk level, and SHAP-based explanation.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build transaction dict
    txn_dict = txn.model_dump()

    # Score
    probability, decision, risk_level, latency_ms = model_service.predict(txn_dict)

    # Explain
    features = model_service.get_preprocessed_features(txn_dict)
    explanation = explainer_service.explain(features, top_k=5)

    # Build response
    top_features = [
        FeatureExplanation(**feat)
        for feat in explanation.get("top_features", [])
    ]

    # Log to database
    log_prediction(
        amount=txn.amount,
        merchant_category=txn.merchant_category,
        hour_of_day=txn.hour_of_day,
        is_foreign=txn.is_foreign,
        distance_from_home=txn.distance_from_home,
        device_trust_score=txn.device_trust_score,
        fraud_probability=probability,
        decision=decision,
        risk_level=risk_level,
        latency_ms=latency_ms,
        explanation_text=explanation.get("explanation_text", ""),
    )

    return ScoringResponse(
        transaction_id=str(uuid.uuid4())[:8],
        fraud_probability=round(probability, 4),
        decision=decision,
        risk_level=risk_level,
        explanation=top_features,
        explanation_text=explanation.get("explanation_text", ""),
        latency_ms=round(latency_ms, 2),
    )


@router.post("/batch", response_model=BatchScoringResponse)
async def score_batch(batch: BatchScoringRequest):
    """Score multiple transactions in one request."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    total_latency = 0.0

    for txn in batch.transactions:
        txn_dict = txn.model_dump()
        probability, decision, risk_level, latency_ms = model_service.predict(txn_dict)
        total_latency += latency_ms

        features = model_service.get_preprocessed_features(txn_dict)
        explanation = explainer_service.explain(features, top_k=5)

        top_features = [
            FeatureExplanation(**feat)
            for feat in explanation.get("top_features", [])
        ]

        results.append(ScoringResponse(
            transaction_id=str(uuid.uuid4())[:8],
            fraud_probability=round(probability, 4),
            decision=decision,
            risk_level=risk_level,
            explanation=top_features,
            explanation_text=explanation.get("explanation_text", ""),
            latency_ms=round(latency_ms, 2),
        ))

    return BatchScoringResponse(
        results=results,
        total_scored=len(results),
        total_latency_ms=round(total_latency, 2),
    )
