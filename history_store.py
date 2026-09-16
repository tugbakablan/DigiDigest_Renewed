from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd


def initialize_history(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                protein_name TEXT NOT NULL,
                sequence TEXT NOT NULL,
                region_start INTEGER NOT NULL,
                region_end INTEGER NOT NULL,
                target_stage TEXT NOT NULL,
                fragments_json TEXT NOT NULL,
                ranked_json TEXT NOT NULL
            )
        """)


def save_analysis(database: Path, protein_name: str, sequence: str,
                  region_start: int, region_end: int, target_stage: str,
                  fragments: pd.DataFrame, ranked: pd.DataFrame) -> int:
    initialize_history(database)
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(database) as connection:
        cursor = connection.execute(
            """INSERT INTO analyses
            (created_at, protein_name, sequence, region_start, region_end, target_stage,
             fragments_json, ranked_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (created_at, protein_name, sequence, region_start, region_end, target_stage,
             fragments.to_json(orient="table", index=False),
             ranked.to_json(orient="table", index=False)),
        )
        return int(cursor.lastrowid)


def list_analyses(database: Path) -> pd.DataFrame:
    initialize_history(database)
    with sqlite3.connect(database) as connection:
        return pd.read_sql_query("""
            SELECT id, created_at, protein_name, length(sequence) AS protein_length,
                   region_start, region_end, target_stage
            FROM analyses ORDER BY id DESC
        """, connection)


def load_analysis(database: Path, analysis_id: int) -> dict:
    initialize_history(database)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        raise ValueError("The saved analysis could not be found.")
    result = dict(row)
    result["fragments"] = pd.read_json(StringIO(result.pop("fragments_json")), orient="table")
    result["ranked"] = pd.read_json(StringIO(result.pop("ranked_json")), orient="table")
    return result


def delete_analysis(database: Path, analysis_id: int) -> None:
    initialize_history(database)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
