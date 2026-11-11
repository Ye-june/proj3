import sqlite3
from contextlib import closing
from pathlib import Path


def record_query(db_path: Path, timestamp: str, response_ms: float) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                response_ms REAL NOT NULL
            )
        """)
        db.execute(
            "INSERT INTO usage_logs (timestamp, response_ms) VALUES (?, ?)",
            (timestamp, response_ms),
        )
        db.commit()

def read_usage(db_path: Path) -> tuple[int, list[tuple[str, float]]]:
    if not db_path.exists():
        return 0, []

    with closing(sqlite3.connect(db_path)) as db:
        total = db.execute("SELECT COUNT(*) FROM usage_logs").fetchone()[0]
        recent = db.execute("""
            SELECT timestamp, response_ms
            FROM usage_logs
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()

    return total, recent