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
    assert "token_store_entries" in table_names


def test_alembic_upgrade_handles_legacy_baseline_and_applies_new_migrations(tmp_path: Path):
    database_path = tmp_path / "legacy-migration-smoke.db"

    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE history (id TEXT PRIMARY KEY, created_at BIGINT NOT NULL)")
        connection.execute(
            """
            CREATE TABLE history_items (
                id TEXT PRIMARY KEY,
                history_id TEXT NOT NULL,
                created_at BIGINT NOT NULL,
                kind TEXT NOT NULL,
                content JSON NOT NULL,
                FOREIGN KEY(history_id) REFERENCES history(id)
            )
            """
        )

    run_migrations(f"sqlite+aiosqlite:///{database_path}")

    with sqlite3.connect(database_path) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        versions = connection.execute("SELECT version_num FROM alembic_version").fetchall()

    table_names = {row[0] for row in tables}
    assert "token_store_entries" in table_names
    assert versions == [("0002_token_store_entries",)]
