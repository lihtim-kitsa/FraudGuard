# FraudGuard — Real-Time Fraud Detection System

An end-to-end machine learning system that scores financial transactions for fraud risk in real time. Features a trained XGBoost/LightGBM model, SHAP explainability, cost-aware decisioning, a FastAPI serving layer, and a live React monitoring dashboard.

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi)
![React](https://img.shields.io/badge/React-19+-blue?logo=react)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

---

## Problem Framing

Credit card fraud costs the global economy **$32+ billion annually**. The challenge isn't just detecting fraud — it's detecting it **in real time** while balancing two competing business objectives:

| Goal | Risk |
|---|---|
| **Catch fraud** (maximize recall) | Blocking legitimate customers → revenue loss, churn |
| **Minimize false alarms** (maximize precision) | Missing fraud → direct financial loss, regulatory fines |

**Why accuracy is the wrong metric:** With ~1.7% fraud rate, a model that predicts "not fraud" for everything achieves 98.3% accuracy. Useless. We optimize for **PR-AUC** (Precision-Recall Area Under Curve) and use a **cost-sensitive threshold** where missing a fraud costs 50× more than a false alarm.

---

## Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   React      │────▶│   FastAPI (uvicorn)   │────▶│  XGBoost Model  │
│   Dashboard  │◀────│   /score /threshold   │◀────│  + SHAP Explainer│
│   (Recharts) │     │   /monitoring         │     │  + Preprocessor  │
└──────────────┘     └──────────┬───────────┘     └─────────────────┘
                                │
                     ┌──────────▼───────────┐
                     │   SQLite (predictions │
                     │   audit log)          │
                     └──────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python -m src.pipeline
```
This will:
- Generate 100K synthetic transactions (~1.7% fraud rate)
- Split data chronologically (70/15/15)
- Train 4 models: Logistic Regression, XGBoost (weighted), XGBoost (SMOTE), LightGBM
- Evaluate on PR-AUC and select the best model
- Create SHAP explainer for feature explanations
- Save all artifacts to `/models`

### 3. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### 4. Start the Dashboard
```bash
cd dashboard
npm install
npm run dev
```
Dashboard: http://localhost:3000

### Docker (alternative)
```bash
docker-compose up --build
```

---

## Results

| Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | ~0.85 | ~0.80 | ~0.75 | ~0.77 |
| **XGBoost (class weighted)** | **~0.96** | **~0.92** | **~0.89** | **~0.90** |
| XGBoost (SMOTE) | ~0.94 | ~0.88 | ~0.87 | ~0.87 |
| LightGBM (class weighted) | ~0.95 | ~0.91 | ~0.88 | ~0.89 |

*Results on the held-out test set (15% of data, chronological split). Exact numbers depend on synthetic data generation.*

### Threshold Decision

We use a **cost-sensitive optimal threshold** instead of the default 0.5:
- **FN cost** (missed fraud): $50 per incident
- **FP cost** (false alarm): $1 per incident
- The optimal threshold minimizes `50 × FN + 1 × FP`

This typically yields a threshold around **0.20–0.35**, favoring recall over precision — better to flag a legitimate transaction for review than to miss actual fraud.

---

## Project Structure

```
fraudguard/
├── src/                      # Core ML pipeline
│   ├── config.py             # Central configuration
│   ├── pipeline.py           # Master pipeline runner
│   ├── data/
│   │   ├── generate_synthetic.py  # Synthetic data generator
│   │   ├── loader.py         # Dataset loading & validation
│   │   └── splitter.py       # Time-aware train/val/test split
│   ├── features/
│   │   └── engineering.py    # Feature engineering pipeline
│   ├── models/
│   │   ├── train.py          # Multi-model training + MLflow
│   │   ├── evaluate.py       # PR-AUC evaluation & reports
│   │   └── explain.py        # SHAP explainability
│   └── utils/
│       └── metrics.py        # Cost-sensitive metrics
├── api/                      # FastAPI serving layer
│   ├── main.py               # App entry point + lifespan
│   ├── schemas.py            # Pydantic request/response models
│   ├── database.py           # SQLite prediction logging
│   ├── routes/
│   │   ├── scoring.py        # POST /score — real-time inference
│   │   ├── threshold.py      # GET/PUT /threshold — business tuning
│   │   └── monitoring.py     # GET /metrics, /drift, /predictions
│   └── services/
│       ├── model_service.py      # Model loading & inference
│       ├── explainer_service.py  # SHAP explanations
│       └── monitor_service.py    # Metrics & drift data
├── dashboard/                # React + Vite + Recharts
│   └── src/
│       ├── App.jsx           # Main app with tabbed navigation
│       ├── components/       # Dashboard components
│       └── hooks/            # API integration
├── models/                   # Saved model artifacts
├── data/                     # Raw & processed datasets
├── mlruns/                   # MLflow experiment logs
├── Dockerfile                # Multi-stage production build
├── docker-compose.yml        # Full stack orchestration
└── requirements.txt          # Python dependencies
```

---

## Key Design Decisions

### 1. Imbalanced Data Handling
- **Class weighting** (`scale_pos_weight`): Adjusts the loss function to penalize minority class errors more heavily. Simple, no data augmentation artifacts.
- **SMOTE comparison**: Generates synthetic minority samples. We compare both approaches empirically.
- **PR-AUC over ROC-AUC**: ROC-AUC can be misleadingly high with imbalanced data. PR-AUC focuses on the minority class performance.

### 2. Time-Aware Splitting
No random shuffling. Data is split chronologically to prevent future information leaking into training. This simulates real deployment where you train on historical data and predict future transactions.

### 3. Cost-Aware Decisioning
Instead of a fixed 0.5 threshold, we optimize for **business cost**:
- Missing fraud (FN) costs 50× more than a false alarm (FP)
- The optimal threshold is found by sweeping all values and minimizing total cost
- Three decision zones: Approve (low risk) → Review (medium) → Decline (high)

### 4. SHAP Explainability
Every prediction comes with a human-readable explanation of *why* it was flagged. Uses `TreeExplainer` for XGBoost/LightGBM (exact, fast). Critical for regulatory compliance and analyst trust.

### 5. Feature Engineering
- **Transaction velocity**: Count of transactions in last 1h/24h window
- **Time since last transaction**: Rapid-fire transactions are suspicious
- **Amount z-scores**: How unusual is this amount for this merchant category?
- **Cyclical hour encoding**: sin/cos transform preserves 23:00 → 01:00 proximity
- **Device trust score**: Low-trust devices correlate with fraud

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/score` | Score a transaction → probability + decision + SHAP explanation |
| `POST` | `/score/batch` | Score multiple transactions |
| `GET` | `/threshold` | Get current threshold + metrics |
| `PUT` | `/threshold` | Update decision thresholds |
| `GET` | `/threshold/sweep` | Metrics at every 0.01 threshold (for PR curve) |
| `GET` | `/monitoring/metrics` | Current model performance |
| `GET` | `/monitoring/feature-importance` | Global SHAP feature ranking |
| `GET` | `/monitoring/drift` | Feature drift over time |
| `GET` | `/monitoring/predictions/recent` | Last 50 scored transactions |
| `GET` | `/health` | Service health check |

---

## Experiment Tracking (MLflow)

All training runs are logged to MLflow:
```bash
mlflow ui --backend-store-uri file:///./mlruns
```
Navigate to http://localhost:5000 to compare models, view parameters, and review metrics.

---

## Dashboard Features

1. **Live Scoring**: Input transaction features → get instant fraud probability with SHAP explanation
2. **Threshold Tuning**: Drag the slider to see precision/recall/cost tradeoffs in real time
3. **Confusion Matrix**: Interactive 2×2 matrix that updates with threshold changes
4. **Feature Importance**: Global SHAP importance bar chart
5. **Drift Monitor**: Simulated feature distribution drift over time windows

---

## FAQ

1. **"Why PR-AUC over accuracy?"** — With 1.7% fraud rate, accuracy is meaningless. PR-AUC evaluates the model specifically on the minority class.
2. **"How do you handle imbalance?"** — Compared class weighting vs. SMOTE empirically. Class weighting won — simpler, no synthetic artifacts.
3. **"How did you choose the threshold?"** — Cost-sensitive optimization. A missed fraud costs 50× more than a false alarm. The optimal threshold minimizes total business cost.
4. **"Can you explain a prediction?"** — Every prediction includes SHAP values showing which features pushed the score up or down.
5. **"How would you monitor this in production?"** — Track feature drift (PSI/KS tests), prediction distribution shifts, and precision/recall on labeled feedback.

---

## License

MIT
