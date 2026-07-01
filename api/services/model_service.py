"""
Model service — loads and manages model inference.

Handles:
- One-time model loading at startup
- Thread-safe inference
- Latency tracking
"""
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.config import (
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_ARTIFACT_PATH,
    DEFAULT_DECLINE_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
)
from src.features.engineering import engineer_features, get_feature_columns


class ModelService:
    """Manages model loading, inference, and threshold configuration."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_type: str = "unknown"
        self.decline_threshold: float = DEFAULT_DECLINE_THRESHOLD
        self.review_threshold: float = DEFAULT_REVIEW_THRESHOLD

    def load(self):
        """Load model and preprocessor from disk."""
        if not MODEL_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_ARTIFACT_PATH}. Run the training pipeline first."
            )
        if not PREPROCESSOR_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {PREPROCESSOR_ARTIFACT_PATH}. "
                "Run the training pipeline first."
            )

        self.model = joblib.load(MODEL_ARTIFACT_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
        self.model_type = type(self.model).__name__
        print(f"✓ Model loaded: {self.model_type}")
        print(f"  Decline threshold: {self.decline_threshold}")
        print(f"  Review threshold:  {self.review_threshold}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def predict(self, transaction: dict) -> tuple[float, str, str, float]:
        """
        Score a single transaction.

        Returns
        -------
        tuple of (probability, decision, risk_level, latency_ms)
        """
        start = time.perf_counter()

        # Build a DataFrame from the transaction dict
        df = pd.DataFrame([transaction])

        # Apply feature engineering
        df = engineer_features(df, fit=False)

        # Get features and transform
        feature_cols = get_feature_columns(df)
        X = self.preprocessor.transform(df[feature_cols])

        # Predict
        probability = float(self.model.predict_proba(X)[0, 1])

        # Decision based on thresholds
        decision = self._make_decision(probability)
        risk_level = self._risk_level(probability)

        latency_ms = (time.perf_counter() - start) * 1000
        return probability, decision, risk_level, latency_ms

    def predict_batch(
        self, transactions: list[dict]
    ) -> list[tuple[float, str, str, float]]:
        """Score multiple transactions."""
        return [self.predict(t) for t in transactions]

    def get_preprocessed_features(self, transaction: dict) -> np.ndarray:
        """Get preprocessed feature vector for SHAP explanation."""
        df = pd.DataFrame([transaction])
        df = engineer_features(df, fit=False)
        feature_cols = get_feature_columns(df)
        return self.preprocessor.transform(df[feature_cols])

    def _make_decision(self, probability: float) -> str:
        """Map probability to business decision."""
        if probability >= self.decline_threshold:
            return "decline"
        elif probability >= self.review_threshold:
            return "review"
        else:
            return "approve"

    def _risk_level(self, probability: float) -> str:
        """Map probability to risk level."""
        if probability >= 0.8:
            return "critical"
        elif probability >= 0.5:
            return "high"
        elif probability >= 0.2:
            return "medium"
        else:
            return "low"

    def update_thresholds(self, decline: float, review: float):
        """Update decision thresholds."""
        self.decline_threshold = decline
        self.review_threshold = review


# Global singleton
model_service = ModelService()
