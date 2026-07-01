# FraudGuard — Real-Time Fraud Detection System

Build an end-to-end ML system that scores financial transactions for fraud risk in real time, with an API, explainability (SHAP), cost-aware decisioning, and a live monitoring dashboard.

## User Review Required

> [!IMPORTANT]
> **Dataset choice**: I'll use the **Kaggle Credit Card Fraud Dataset** (ULB, 284,807 transactions, 492 frauds). The features are PCA-anonymized (V1–V28) plus `Time` and `Amount`. This is the industry-standard benchmark. Since Kaggle requires authentication to download, **you'll need to either:**
> - Place the `creditcard.csv` file manually into `/data/raw/`, OR
> - Provide Kaggle API credentials so I can download it programmatically
>
> I'll generate a **synthetic fallback dataset** (~100K transactions with realistic features) so the entire pipeline works out of the box without manual downloads.

> [!WARNING]
> **MLflow**: Running MLflow locally requires a running tracking server. I'll configure it to log to a local `mlruns/` directory (file-based, no server needed) so it works without any extra setup.

## Open Questions

> [!IMPORTANT]
> 1. **Synthetic vs. real dataset**: Should I build exclusively around the synthetic dataset (fully self-contained, richer feature names) or primarily target the Kaggle CSV with synthetic as fallback?
> 2. **Docker**: Do you have Docker Desktop installed on Windows? This determines whether I build+test the Docker image or just provide the Dockerfile with documentation.
> 3. **Node.js version**: Do you have Node.js installed? What version? (needed for the React dashboard)

---

## Proposed Changes

The project will live in `c:\Users\astik\OneDrive\Desktop\ML Project` with this structure:

```
fraudguard/
├── data/
│   ├── raw/                    # Raw dataset (creditcard.csv or generated)
│   └── processed/              # Train/val/test splits
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_modeling.ipynb       # Model training & evaluation
├── src/
│   ├── __init__.py
│   ├── config.py               # Central configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate_synthetic.py  # Synthetic dataset generator
│   │   ├── loader.py           # Data loading & validation
│   │   └── splitter.py         # Time-aware train/val/test split
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py      # Feature engineering pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # Training orchestration
│   │   ├── evaluate.py         # Evaluation metrics & reports
│   │   └── explain.py          # SHAP explainability
│   └── utils/
│       ├── __init__.py
│       └── metrics.py          # Custom metrics (PR-AUC, cost-sensitive)
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── schemas.py              # Pydantic request/response models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── scoring.py          # POST /score endpoint
│   │   ├── threshold.py        # GET/PUT /threshold endpoint
│   │   └── monitoring.py       # GET /metrics, /drift endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py    # Model loading & inference
│   │   ├── explainer_service.py # SHAP explanations
│   │   └── monitor_service.py  # Drift & performance tracking
│   └── database.py             # SQLite for logging predictions
├── dashboard/                  # React + Vite + Recharts
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ScoreTransaction.jsx    # Live scoring form
│   │   │   ├── ConfusionMatrix.jsx     # Interactive confusion matrix
│   │   │   ├── ThresholdSlider.jsx     # PR/recall tradeoff control
│   │   │   ├── FeatureImportance.jsx   # SHAP waterfall chart
│   │   │   ├── DriftMonitor.jsx        # Model drift over time
│   │   │   └── MetricsOverview.jsx     # KPI cards
│   │   ├── hooks/
│   │   │   └── useApi.js               # API integration hooks
│   │   └── styles/
│   │       └── index.css               # Design system
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/                     # Saved model artifacts
│   └── .gitkeep
├── mlruns/                     # MLflow experiment logs
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

---

### Phase 1 — Data & Feature Engineering

#### [NEW] [generate_synthetic.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/src/data/generate_synthetic.py)
- Generate ~100K synthetic transactions with realistic fraud patterns
- Features: `amount`, `hour_of_day`, `day_of_week`, `merchant_category`, `is_foreign`, `distance_from_home`, `velocity_24h`, `time_since_last_txn`, `amount_zscore`, `device_trust_score`
- Class imbalance: ~1.7% fraud rate (realistic)
- Temporal ordering preserved for time-aware splitting

#### [NEW] [loader.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/src/data/loader.py)
- Load either real Kaggle CSV or synthetic data
- Schema validation, dtype enforcement, null handling

#### [NEW] [splitter.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/src/data/splitter.py)
- **Time-aware split** (no random shuffle — prevents data leakage)
- 70% train / 15% validation / 15% test by temporal ordering

#### [NEW] [engineering.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/src/features/engineering.py)
- Transaction velocity (count in last 1h, 24h)
- Time-since-last-transaction
- Amount z-scores (per-merchant-category rolling)
- Hour/day cyclical encoding (sin/cos)
- Log-transform skewed amounts

---

### Phase 2 — Modeling & Evaluation

#### [NEW] [train.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/src/models/train.py)
- **Baseline**: Logistic Regression with class weighting
- **Main models**: XGBoost and LightGBM with:
  - `scale_pos_weight` / class weighting
  - SMOTE comparison (via `imbalanced-learn`)
- All experiments logged to MLflow (local file store)
- Hyperparameter tuning via Optuna or GridSearchCV

#### [NEW] [evaluate.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/src/models/evaluate.py)
- Primary metric: **PR-AUC** (not ROC-AUC — critical for imbalanced data)
- Also track: F1, Precision, Recall, FPR at various thresholds
- Cost-sensitive evaluation: configurable FP/FN cost ratio
- Generate confusion matrices, PR curves, threshold-vs-metrics plots
- Save evaluation report as JSON + plots

#### [NEW] [explain.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/src/models/explain.py)
- `TreeExplainer` for XGBoost/LightGBM (fast, exact)
- Global feature importance (mean |SHAP|)
- Per-transaction explanation (top-5 contributing features)

#### [NEW] [metrics.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/src/utils/metrics.py)
- Custom cost-weighted loss function
- Optimal threshold finder (minimize total business cost)
- Threshold sweep analysis

---

### Phase 3 — API (FastAPI)

#### [NEW] [main.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/api/main.py)
- FastAPI app with lifespan (model loaded once at startup)
- CORS middleware for dashboard
- Health check endpoint `/health`

#### [NEW] [schemas.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/api/schemas.py)
- `TransactionRequest`: Pydantic model matching feature schema
- `ScoringResponse`: probability, decision (approve/review/decline), SHAP explanation
- `ThresholdConfig`: adjustable threshold + cost parameters
- `MetricsResponse`: current model performance stats

#### [NEW] [scoring.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/api/routes/scoring.py)
- `POST /score` — score single transaction (<200ms target)
  - Returns: `fraud_probability`, `decision`, `risk_level`, `explanation` (top SHAP features)
- `POST /score/batch` — score multiple transactions

#### [NEW] [threshold.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/api/routes/threshold.py)
- `GET /threshold` — current threshold + resulting metrics
- `PUT /threshold` — update threshold, returns new precision/recall/F1
- `GET /threshold/sweep` — returns metrics at every 0.01 increment

#### [NEW] [monitoring.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/api/routes/monitoring.py)
- `GET /metrics` — current model metrics on test set
- `GET /drift` — simulated feature drift over time windows
- `GET /predictions/recent` — last N scored transactions from SQLite log

#### [NEW] [model_service.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/api/services/model_service.py)
- Loads serialized model + preprocessor at startup
- Thread-safe inference with timing

#### [NEW] [explainer_service.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/api/services/explainer_service.py)
- SHAP TreeExplainer initialized once
- Returns human-readable explanations: `"High transaction amount: +0.23"`

#### [NEW] [database.py](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/api/database.py)
- SQLite database for prediction logging
- Table: `predictions` (timestamp, features, score, decision, latency_ms)

---

### Phase 4 — React Dashboard

#### Technology
- **Vite** + React (fast dev server)
- **Recharts** for all charts
- **Vanilla CSS** with a premium dark-mode design system

#### [NEW] Dashboard Components

| Component | Purpose |
|---|---|
| `ScoreTransaction.jsx` | Form to input transaction features, calls `/score`, shows result with SHAP explanation |
| `ThresholdSlider.jsx` | Draggable slider that calls `/threshold/sweep`, shows live PR curve & business cost |
| `ConfusionMatrix.jsx` | 2×2 heatmap that updates dynamically with threshold changes |
| `FeatureImportance.jsx` | Horizontal bar chart of global SHAP importance |
| `DriftMonitor.jsx` | Time-series line chart showing feature distribution drift |
| `MetricsOverview.jsx` | KPI cards: PR-AUC, F1, precision, recall, total transactions scored |

#### Design Aesthetic
- **Dark mode** with deep navy (`#0a0e1a`) background
- **Accent gradients**: electric blue → cyan (`#3b82f6` → `#06b6d4`)
- **Glassmorphism** cards with `backdrop-filter: blur(16px)`
- **Google Fonts**: Inter for body, JetBrains Mono for metrics
- Smooth transitions on all interactive elements
- Responsive grid layout (works on 1200px+ screens)

---

### Phase 5 — Containerization & Documentation

#### [NEW] [Dockerfile](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/Dockerfile)
- Multi-stage build:
  - Stage 1: Python dependencies + model artifacts
  - Stage 2: Node build for dashboard static assets
  - Stage 3: Slim production image serving both API + static dashboard

#### [NEW] [docker-compose.yml](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/docker-compose.yml)
- `api` service (FastAPI on port 8000)
- `dashboard` service (Vite dev or Nginx on port 3000)

#### [NEW] [README.md](file:///c:/Users/astik/OneDrive/Desktop/ML%20Project/README.md)
- Business framing & problem description
- Architecture diagram (Mermaid)
- Quick start (local + Docker)
- Results: PR-AUC, threshold reasoning, cost tradeoff
- Interview talking points

---

## Verification Plan

### Automated Tests
```bash
# Unit tests for feature engineering
python -m pytest tests/test_features.py -v

# Model evaluation sanity checks
python -m pytest tests/test_model.py -v

# API endpoint tests
python -m pytest tests/test_api.py -v

# API response time validation (<200ms)
python -c "import requests, time; t=time.time(); requests.post('http://localhost:8000/score', json={...}); print(f'{(time.time()-t)*1000:.0f}ms')"
```

### Manual Verification
- Run the full pipeline end-to-end: generate data → train → evaluate → serve
- Open dashboard at `http://localhost:3000` and verify:
  - Live scoring returns results with SHAP explanations
  - Threshold slider updates confusion matrix in real time
  - Feature importance chart renders correctly
  - Drift monitoring shows simulated drift data
- Verify Docker build completes and containers run

## Execution Order

1. Project scaffolding (config, requirements, .gitignore)
2. Synthetic data generator + data loading pipeline
3. Feature engineering
4. Model training + evaluation + SHAP
5. FastAPI with all endpoints
6. SQLite logging
7. React dashboard (Vite + Recharts)
8. Docker + README
9. End-to-end verification
