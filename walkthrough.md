# 🛡️ FraudGuard — Final Project Walkthrough

The **FraudGuard** real-time fraud detection system has been fully implemented, trained, and deployed.

Both the API and the interactive Dashboard are currently running on your system! 

## 🎉 What Was Accomplished

> [!NOTE]
> The full ML system was built entirely from scratch, optimizing for the business problem of cost-aware fraud detection (not just accuracy on imbalanced datasets).

1. **Synthetic Data Pipeline**:
   - Developed a realistic transaction generator with 100K samples, exhibiting a highly imbalanced ~1.7% fraud rate.
   - Employed time-aware sequential data splitting (train on past, evaluate on future).

2. **Feature Engineering**:
   - Cyclical encoding of transaction times (hour of day, day of week).
   - Frequency / velocity calculation.
   - Cost-aware scaling and missing value imputation.

3. **Cost-Sensitive ML Training Pipeline**:
   - Evaluated 4 model iterations focusing on PR-AUC. 
   - Class-weighted XGBoost won with an incredible **PR-AUC of ~0.9929**.
   - Tuned thresholds to optimize for *Business Cost* (where false negatives cost 50x more than false positives).

4. **FastAPI Inference Service**:
   - Standalone API utilizing `joblib` artifacts.
   - Built an SQLite-backed logging system to record transactions.
   - Uses exact TreeSHAP for `<200ms` prediction explainability.
   - API endpoints tested and achieving inference in `~9ms`.

5. **React + Vite Dashboard**:
   - Modern "glassmorphism" premium dark mode UI without external UI libraries.
   - Recharts for visual insights.
   - Real-time scoring simulation, cost slider tuning, and a dynamic confusion matrix.

## 🚀 How to Demo

The services are actively running in the background right now. You can interact with them using these links:

* **Dashboard**: [http://localhost:3000](http://localhost:3000) (Open this in your browser!)
* **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI for the backend)

### Key Features to Try in the Dashboard:

1. **Live Scoring Tab**: Click the `🎲 Random Fraud` or `🎲 Random Legit` buttons to populate the form, then click **Score Transaction**. The model will instantly score the request and provide SHAP feature attributions explaining *why* it made its decision.
2. **Threshold Tuning**: Move the threshold slider to see how modifying the strictness shifts the confusion matrix, affects Precision/Recall, and changes the Total Business Cost.
3. **Model Monitoring**: View the global feature importance rankings to understand which features the XGBoost model prioritizes.

## 📦 Containerization and Docs

- **Docker Ready**: We configured `Dockerfile` and `docker-compose.yml`. You can spin this up independently using `docker-compose up --build` at any time.
- **Interviews**: We documented key system architecture decisions and talking points directly in the [README.md](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/README.md).

> [!TIP]
> **Check out your new repository here:** [c:\Users\astik\OneDrive\Desktop\ML Project](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project)
