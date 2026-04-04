import asyncio
from pathlib import Path
from shutil import rmtree

from sqlalchemy import delete

from src.config.factory import get_config
from src.core.database import get_engine, get_session, run_migrations
from src.history.repo.async_sqlalchemy.models import HistoryDb, HistoryItemDb

# Order matters because of foreign key constraints
DBMODELS_TO_DELETE = [HistoryItemDb, HistoryDb]


async def reset_database():
    database_connection_string = get_config().database.connection_string
    run_migrations(database_connection_string)
    engine = get_engine(database_connection_string)

    async with get_session(engine) as session:
        for model in DBMODELS_TO_DELETE:
            stmt = delete(model)
            await session.execute(stmt)


def reset_rag_database():
    path = Path("./data/qdrant_storage/collections")
    rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


async def main():
    response = input("Are you sure you want to delete the database? (y/): ")
    if response != "y":
        print("Aborting...")
        exit(1)

    await reset_database()
    reset_rag_database()


if __name__ == "__main__":
    asyncio.run(main())
