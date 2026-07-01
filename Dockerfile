# ── Stage 1: Python backend ─────────────────────────────────────
FROM python:3.12-slim AS backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY api/ api/

# Copy model artifacts (must be pre-trained)
COPY models/ models/

# ── Stage 2: Node frontend build ────────────────────────────────
FROM node:20-alpine AS dashboard-build

WORKDIR /dashboard

COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm install

COPY dashboard/ .
RUN npm run build

# ── Stage 3: Production image ───────────────────────────────────
FROM python:3.12-slim AS production

WORKDIR /app

# Install runtime dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY --from=backend /app/src/ src/
COPY --from=backend /app/api/ api/
COPY --from=backend /app/models/ models/

# Copy built dashboard
COPY --from=dashboard-build /dashboard/dist/ dashboard/dist/

# Create data directories
RUN mkdir -p data/raw data/processed mlruns

EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
