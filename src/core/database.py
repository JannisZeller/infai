from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlmodel import SQLModel

SessionContext = AsyncGenerator[AsyncSession, None]


def get_engine(path: Path = Path("./data/database.db")) -> AsyncEngine:
    return create_async_engine(
        f"sqlite+aiosqlite:///{path}",
        echo=False,
        pool_pre_ping=True,  # Verify connections are alive before using
        pool_recycle=3600,  # Recycle connections after 1 hour
        connect_args={"timeout": 30},  # Connection timeout
    )


async def create_db_and_tables(engine: AsyncEngine):
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


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
