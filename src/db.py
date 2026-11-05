"""
Database persistence layer for storing and querying light curve analysis records.
"""
import sqlite3
from datetime import datetime, timezone

from src import PROJECT_ROOT

DB_PATH = str(PROJECT_ROOT / "regulus.db")


def init_db() -> None:
    """Initialize SQLite database tables if they do not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                prediction TEXT,
                confidence REAL,
                transit_depth REAL,
                period REAL,
                bls_power REAL,
                created_at TEXT
            )
        """)
        conn.commit()


def save_analysis(filename: str, result: dict) -> None:
    """
    Persist an analysis result to the database.

    Args:
        filename: Source file name or sample identifier.
        result: Prediction dictionary containing classification and transit metrics.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    prediction = str(result.get("prediction", "unknown"))
    confidence = float(result.get("confidence", 0.0))
    transit_depth = float(result.get("depth", 0.0))
    period = float(result.get("period", 0.0))
    bls_power = float(result.get("bls_power", 0.0))

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO analyses (
                filename, prediction, confidence, transit_depth, period, bls_power, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (filename, prediction, confidence, transit_depth, period, bls_power, created_at),
        )
        conn.commit()


def load_history() -> list[dict]:
    """
    Fetch all past analysis records ordered by creation timestamp descending.

    Returns:
        List of dictionaries representing individual analysis runs.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, filename, prediction, confidence, transit_depth, period, bls_power, created_at
            FROM analyses
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def clear_history() -> None:
    """Clear all analysis records from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM analyses")
        conn.commit()
