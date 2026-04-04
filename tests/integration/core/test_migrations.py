import sqlite3
from pathlib import Path

from src.core.database import run_migrations


def test_alembic_upgrade_creates_history_tables(tmp_path: Path):
    database_path = tmp_path / "migration-smoke.db"
    run_migrations(f"sqlite+aiosqlite:///{database_path}")

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    table_names = {row[0] for row in rows}
    assert "history" in table_names
    assert "history_items" in table_names
