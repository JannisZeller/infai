import asyncio
from pathlib import Path
from shutil import rmtree

from sqlalchemy import delete

from src.core.database import create_db_and_tables, get_engine, get_session
from src.history.async_sqlalchemy.models import HistoryDb, HistoryItemDb

# Order matters because of foreign key constraints
DBMODELS_TO_DELETE = [HistoryItemDb, HistoryDb]


async def reset_database():
    engine = get_engine()
    await create_db_and_tables(engine)

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
