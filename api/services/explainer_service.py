"""
Explainer service — wraps SHAP for API usage.

Loads the SHAP TreeExplainer once and provides per-transaction explanations.
"""
import joblib
from pathlib import Path

from src.config import SHAP_EXPLAINER_PATH, PREPROCESSOR_ARTIFACT_PATH
from src.models.explain import explain_single, get_feature_names, global_feature_importance


class ExplainerService:
    """Manages SHAP explanations for the API."""

    def __init__(self):
        self.explainer = None
        self.feature_names: list[str] = []

    def load(self):
        """Load the SHAP explainer and feature names."""
        if not SHAP_EXPLAINER_PATH.exists():
            print("⚠ SHAP explainer not found — explanations will be unavailable")
            return

        self.explainer = joblib.load(SHAP_EXPLAINER_PATH)
        preprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
        self.feature_names = get_feature_names(preprocessor)
        print(f"✓ SHAP explainer loaded ({len(self.feature_names)} features)")

    @property
    def is_loaded(self) -> bool:
        return self.explainer is not None

    def explain(self, features, top_k: int = 5) -> dict:
        """
        Explain a single prediction.

        Parameters
        ----------
        features : np.ndarray
            Preprocessed feature vector.
        top_k : int
            Number of top features to return.

        Returns
        -------
        dict with top_features, base_value, explanation_text
        """
        if not self.is_loaded:
            return {
                "top_features": [],
                "base_value": 0.0,
                "explanation_text": "Explainer not available",
            }

        return explain_single(
            features,
            explainer=self.explainer,
            feature_names=self.feature_names,
            top_k=top_k,
        )


# Global singleton
explainer_service = ExplainerService()
