from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from alembic import command
from alembic.config import Config as AlembicConfig

SessionContext = AsyncGenerator[AsyncSession, None]
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///data/database.db"
ALEMBIC_CONFIG_PATH = Path(__file__).resolve().parents[2] / "alembic.ini"
LEGACY_BASELINE_TABLES = {"history", "history_items"}
LEGACY_BASELINE_REVISION = "0001_history_tables"


def get_engine(database_url: str = DEFAULT_DATABASE_URL) -> AsyncEngine:
    return create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,  # Verify connections are alive before using
        pool_recycle=3600,  # Recycle connections after 1 hour
        connect_args={"timeout": 30},  # Connection timeout
    )


async def create_db_and_tables(engine: AsyncEngine):
    raise RuntimeError("create_db_and_tables has been replaced by Alembic migrations; use run_migrations instead.")


def _to_sync_database_url(database_url: str) -> str:
    return database_url.replace("+aiosqlite", "")


def run_migrations(database_url: str = DEFAULT_DATABASE_URL, revision: str = "head") -> None:
    alembic_config = AlembicConfig(str(ALEMBIC_CONFIG_PATH))
    sync_database_url = _to_sync_database_url(database_url)
    alembic_config.set_main_option("sqlalchemy.url", sync_database_url)

    engine = create_engine(sync_database_url)
    with engine.begin() as connection:
        inspector = inspect(connection)
        existing_tables = set(inspector.get_table_names())
        has_empty_alembic_version = False
        if "alembic_version" in existing_tables:
            version_rows = connection.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
            has_empty_alembic_version = version_rows.scalar_one_or_none() is None

    if LEGACY_BASELINE_TABLES.issubset(existing_tables) and (
        "alembic_version" not in existing_tables or has_empty_alembic_version
    ):
        command.stamp(alembic_config, LEGACY_BASELINE_REVISION)
        command.upgrade(alembic_config, revision)
        return

    command.upgrade(alembic_config, revision)


@asynccontextmanager
async def get_session(engine: AsyncEngine) -> SessionContext:
    session = AsyncSession(
        bind=engine,
        autobegin=False,
        autocommit=False,
    )
    try:
        await session.begin()
        # Enable foreign key constraints in SQLite for each session
        if engine.dialect.name == "sqlite":
            await session.execute(text("PRAGMA foreign_keys = ON"))
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise e
    finally:
        await session.close()
