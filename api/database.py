"""
SQLite database for logging predictions.

Stores every scored transaction for monitoring, drift detection,
and audit trail purposes.
"""
import sqlite3
import datetime
from pathlib import Path
from src.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection, creating the DB if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the predictions table if it doesn't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            amount REAL,
            merchant_category TEXT,
            hour_of_day REAL,
            is_foreign INTEGER,
            distance_from_home REAL,
            device_trust_score REAL,
            fraud_probability REAL NOT NULL,
            decision TEXT NOT NULL,
            risk_level TEXT,
            latency_ms REAL,
            explanation_text TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(
    amount: float,
    merchant_category: str,
    hour_of_day: float,
    is_foreign: int,
    distance_from_home: float,
    device_trust_score: float,
    fraud_probability: float,
    decision: str,
    risk_level: str,
    latency_ms: float,
    explanation_text: str,
):
    """Log a single prediction to the database."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO predictions
        (timestamp, amount, merchant_category, hour_of_day, is_foreign,
         distance_from_home, device_trust_score, fraud_probability, decision,
         risk_level, latency_ms, explanation_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            amount,
            merchant_category,
            hour_of_day,
            is_foreign,
            distance_from_home,
            device_trust_score,
            fraud_probability,
            decision,
            risk_level,
            latency_ms,
            explanation_text,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit: int = 50) -> list[dict]:
    """Fetch the most recent predictions."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_prediction_count() -> int:
    """Get total number of predictions."""
    conn = get_connection()
    cursor = conn.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_prediction_stats() -> dict:
    """Get aggregate statistics from prediction logs."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            AVG(fraud_probability) as avg_score,
            AVG(latency_ms) as avg_latency,
            SUM(CASE WHEN decision = 'decline' THEN 1 ELSE 0 END) as decline_count,
            SUM(CASE WHEN decision = 'review' THEN 1 ELSE 0 END) as review_count,
            SUM(CASE WHEN decision = 'approve' THEN 1 ELSE 0 END) as approve_count
        FROM predictions
    """)
    row = dict(cursor.fetchone())
    conn.close()
    return row
