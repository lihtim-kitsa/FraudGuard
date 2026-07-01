"""
Pydantic schemas for the FraudGuard API.

Defines request/response models for all endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """Single transaction to score for fraud risk."""
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    merchant_category: str = Field(
        ...,
        description="Merchant category (grocery, gas_station, restaurant, "
                    "online_retail, travel, entertainment, healthcare, "
                    "electronics, jewelry, cash_advance)"
    )
    hour_of_day: float = Field(..., ge=0, lt=24, description="Hour of transaction (0-23.99)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    is_foreign: int = Field(..., ge=0, le=1, description="1 if foreign transaction, 0 otherwise")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in km")
    device_trust_score: float = Field(..., ge=0, le=1, description="Device trust score (0-1)")
    timestamp: Optional[float] = Field(None, description="Unix timestamp (optional)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "amount": 245.50,
                    "merchant_category": "electronics",
                    "hour_of_day": 2.5,
                    "day_of_week": 5,
                    "is_foreign": 1,
                    "distance_from_home": 120.0,
                    "device_trust_score": 0.3,
                }
            ]
        }
    }


class FeatureExplanation(BaseModel):
    """SHAP explanation for a single feature."""
    feature: str
    shap_value: float
    direction: str


class ScoringResponse(BaseModel):
    """Response from the scoring endpoint."""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    decision: str = Field(..., description="approve | review | decline")
    risk_level: str = Field(..., description="low | medium | high | critical")
    explanation: list[FeatureExplanation]
    explanation_text: str
    latency_ms: float


class BatchScoringRequest(BaseModel):
    """Batch of transactions to score."""
    transactions: list[TransactionRequest]


class BatchScoringResponse(BaseModel):
    """Response from the batch scoring endpoint."""
    results: list[ScoringResponse]
    total_scored: int
    total_latency_ms: float


# ──────────────────────────────────────────────
# Threshold
# ──────────────────────────────────────────────
class ThresholdConfig(BaseModel):
    """Current threshold configuration."""
    decline_threshold: float = Field(..., ge=0, le=1)
    review_threshold: float = Field(..., ge=0, le=1)


class ThresholdResponse(BaseModel):
    """Threshold + resulting metrics."""
    decline_threshold: float
    review_threshold: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    total_cost: float
    tp: int
    fp: int
    tn: int
    fn: int


class ThresholdSweepPoint(BaseModel):
    """Single point in the threshold sweep."""
    threshold: float
    precision: float
    recall: float
    f1: float
    total_cost: float
    tp: int
    fp: int
    tn: int
    fn: int


# ──────────────────────────────────────────────
# Monitoring
# ──────────────────────────────────────────────
class ModelMetrics(BaseModel):
    """Current model performance metrics."""
    model_type: str
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    total_predictions: int
    fraud_rate: float


class FeatureImportanceItem(BaseModel):
    """Single feature importance entry."""
    feature: str
    importance: float


class DriftPoint(BaseModel):
    """Single drift measurement point."""
    window: str
    feature: str
    mean_shift: float
    std_shift: float
    drift_score: float


class PredictionLog(BaseModel):
    """Logged prediction record."""
    id: int
    timestamp: str
    amount: float
    fraud_probability: float
    decision: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: str
    version: str
