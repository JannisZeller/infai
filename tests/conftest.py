from contextlib import asynccontextmanager

from src.core.database import SessionContext, create_db_and_tables, get_engine, get_session


@asynccontextmanager
async def get_test_session() -> SessionContext:
    engine = get_engine()
    await create_db_and_tables(engine)

    async with get_session(engine) as session:
        yield session
