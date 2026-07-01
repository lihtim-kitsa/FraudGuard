"""
FraudGuard — Master pipeline runner.

Runs the full pipeline:
1. Generate / load dataset
2. Time-aware split
3. Feature engineering + preprocessing
4. Train all models
5. Evaluate best model on test set
6. Create SHAP explainer
7. Save all artifacts

Usage:
    python -m src.pipeline
"""
import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_dataset
from src.data.splitter import time_aware_split
from src.features.engineering import engineer_features, build_preprocessor, get_feature_columns
from src.models.train import train_all_models
from src.models.evaluate import evaluate_model
from src.models.explain import create_shap_explainer, global_feature_importance, get_feature_names
from src.config import MODEL_ARTIFACT_PATH, PREPROCESSOR_ARTIFACT_PATH, MODELS_DIR

import joblib
import json


def run_pipeline():
    """Run the full FraudGuard training pipeline."""
    print("=" * 60)
    print("  FraudGuard — Training Pipeline")
    print("=" * 60)

    # ── Step 1: Load data ────────────────────────────────────────
    print("\n📂 Step 1: Loading dataset...")
    df = load_dataset()

    # ── Step 2: Time-aware split ─────────────────────────────────
    print("\n✂️  Step 2: Splitting data (time-aware)...")
    train_df, val_df, test_df = time_aware_split(df)

    # ── Step 3: Train all models ─────────────────────────────────
    print("\n🤖 Step 3: Training models...")
    results = train_all_models(train_df, val_df)

    # ── Step 4: Evaluate best model on test set ──────────────────
    print("\n📈 Step 4: Evaluating on test set...")
    report = evaluate_model(test_df)

    # ── Step 5: Create SHAP explainer ────────────────────────────
    print("\n🔍 Step 5: Creating SHAP explainer...")
    model = joblib.load(MODEL_ARTIFACT_PATH)
    preprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
    explainer = create_shap_explainer(model, preprocessor)

    # Compute global feature importance
    test_engineered = engineer_features(test_df, fit=False)
    feature_cols = get_feature_columns(test_engineered)
    X_test = preprocessor.transform(test_engineered[feature_cols])
    feature_names = get_feature_names(preprocessor)
    importance = global_feature_importance(X_test, explainer, feature_names)

    # Save feature importance
    importance_path = MODELS_DIR / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(importance, f, indent=2)
    print(f"✓ Feature importance saved: {importance_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Best model:     {report['model_type']}")
    print(f"  PR-AUC:         {report['metrics_at_optimal']['pr_auc']:.4f}")
    print(f"  Precision:      {report['metrics_at_optimal']['precision']:.4f}")
    print(f"  Recall:         {report['metrics_at_optimal']['recall']:.4f}")
    print(f"  F1 Score:       {report['metrics_at_optimal']['f1']:.4f}")
    print(f"  Threshold:      {report['optimal_threshold']}")
    print(f"\n  Top 5 features:")
    for feat in importance[:5]:
        print(f"    {feat['feature']:30s}  {feat['importance']:.4f}")

    return report


if __name__ == "__main__":
    run_pipeline()
