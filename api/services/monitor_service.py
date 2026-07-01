"""
Monitor service — provides model metrics, drift simulation, and threshold analysis.

Loads evaluation report data and provides simulated drift monitoring.
"""
import json
import numpy as np
from pathlib import Path

from src.config import EVALUATION_REPORT_PATH, MODELS_DIR


class MonitorService:
    """Provides monitoring data for the dashboard."""

    def __init__(self):
        self.evaluation_report: dict = {}
        self.full_report: dict = {}
        self.feature_importance: list[dict] = []

    def load(self):
        """Load evaluation reports and feature importance."""
        # Load slim report
        if EVALUATION_REPORT_PATH.exists():
            with open(EVALUATION_REPORT_PATH) as f:
                self.evaluation_report = json.load(f)
            print("✓ Evaluation report loaded")

        # Load full report (with y_scores, y_true, threshold_sweep)
        full_path = MODELS_DIR / "evaluation_report_full.json"
        if full_path.exists():
            with open(full_path) as f:
                self.full_report = json.load(f)
            print("✓ Full evaluation report loaded")

        # Load feature importance
        importance_path = MODELS_DIR / "feature_importance.json"
        if importance_path.exists():
            with open(importance_path) as f:
                self.feature_importance = json.load(f)
            print(f"✓ Feature importance loaded ({len(self.feature_importance)} features)")

    def get_metrics(self) -> dict:
        """Get current model metrics."""
        if not self.evaluation_report:
            return {}
        m = self.evaluation_report.get("metrics_at_optimal", {})
        return {
            "model_type": self.evaluation_report.get("model_type", "unknown"),
            "pr_auc": m.get("pr_auc", 0),
            "roc_auc": m.get("roc_auc", 0),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "f1": m.get("f1", 0),
            "threshold": m.get("threshold", 0.5),
            "total_predictions": self.evaluation_report.get("test_size", 0),
            "fraud_rate": (
                self.evaluation_report.get("fraud_count", 0) /
                max(self.evaluation_report.get("test_size", 1), 1)
            ),
        }

    def get_threshold_sweep(self) -> list[dict]:
        """Get the threshold sweep data for the dashboard."""
        if self.full_report and "threshold_sweep" in self.full_report:
            return self.full_report["threshold_sweep"]
        return []

    def get_threshold_metrics(self, threshold: float) -> dict:
        """Compute metrics at a specific threshold using saved predictions."""
        if not self.full_report:
            return {}

        y_true = np.array(self.full_report.get("y_true", []))
        y_scores = np.array(self.full_report.get("y_scores", []))

        if len(y_true) == 0 or len(y_scores) == 0:
            return {}

        from src.utils.metrics import compute_all_metrics
        return compute_all_metrics(y_true, y_scores, threshold=threshold)

    def get_feature_importance(self) -> list[dict]:
        """Get global feature importance."""
        return self.feature_importance

    def get_drift_data(self) -> list[dict]:
        """
        Generate simulated drift data for the dashboard.

        In production, this would compare live feature distributions
        against training data. Here we simulate it for demonstration.
        """
        rng = np.random.default_rng(42)
        features = ["amount", "distance_from_home", "velocity_24h",
                     "device_trust_score", "hour_of_day"]
        windows = ["Week 1", "Week 2", "Week 3", "Week 4",
                    "Week 5", "Week 6", "Week 7", "Week 8"]

        drift_data = []
        for feature in features:
            base_drift = rng.uniform(0.01, 0.05)
            for i, window in enumerate(windows):
                # Simulate gradual drift with some noise
                drift = base_drift * (1 + i * 0.15) + rng.normal(0, 0.01)
                drift_data.append({
                    "window": window,
                    "feature": feature,
                    "mean_shift": round(float(rng.normal(0, drift)), 4),
                    "std_shift": round(float(abs(rng.normal(0, drift * 0.5))), 4),
                    "drift_score": round(float(max(0, drift)), 4),
                })

        return drift_data


# Global singleton
monitor_service = MonitorService()
