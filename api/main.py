"""
FraudGuard API — FastAPI application entry point.

Loads model, SHAP explainer, and evaluation data at startup.
Provides /score, /threshold, /monitoring, and /health endpoints.
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.schemas import HealthResponse
from api.routes import scoring, threshold, monitoring
from api.services.model_service import model_service
from api.services.explainer_service import explainer_service
from api.services.monitor_service import monitor_service
from api.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and services at startup, clean up on shutdown."""
    print("\n🚀 Starting FraudGuard API...")

    # Initialize database
    init_db()

    # Load model and services
    try:
        model_service.load()
        explainer_service.load()
        monitor_service.load()
        print("✓ All services loaded successfully\n")
    except FileNotFoundError as e:
        print(f"⚠ {e}")
        print("  Run `python -m src.pipeline` to train the model first.\n")

    yield

    print("\n🛑 Shutting down FraudGuard API...")


# ── Create FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="FraudGuard API",
    description=(
        "Real-time fraud detection API. Score transactions for fraud risk, "
        "get SHAP explanations, adjust decision thresholds, and monitor "
        "model performance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow dashboard to connect) ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to dashboard origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ──────────────────────────────────────────────
app.include_router(scoring.router)
app.include_router(threshold.router)
app.include_router(monitoring.router)


# ── Health check ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API and model are operational."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        model_type=model_service.model_type,
        version="1.0.0",
    )


@app.get("/", tags=["Root"])
async def root():
    """API root — redirects to docs."""
    return {
        "name": "FraudGuard API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
