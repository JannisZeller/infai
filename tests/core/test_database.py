from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlmodel import col

from src.history.async_sqlalchemy.models import HistoryDb, HistoryItemDb
from tests.conftest import get_test_session

HISTORY_ID = uuid4()


async def reset_database():
    async with get_test_session() as session:
        stmt = delete(HistoryItemDb).where(col(HistoryItemDb.history_id) == HISTORY_ID)
        await session.execute(stmt)
        stmt = delete(HistoryDb).where(col(HistoryDb.id) == HISTORY_ID)
        await session.execute(stmt)


async def test_create_history():
    async with get_test_session() as session:
        history = HistoryDb(id=HISTORY_ID)
        session.add(history)

    async with get_test_session() as session:
        query = select(HistoryDb).where(col(HistoryDb.id) == HISTORY_ID)
        result = await session.execute(query)
        history = result.scalar_one_or_none()
        assert history is not None
        assert history.id == HISTORY_ID

    await reset_database()


async def test_delete_history_with_items_raises_exc():
    async with get_test_session() as session:
        history = HistoryDb(id=HISTORY_ID)
        session.add(history)
        test_item = HistoryItemDb(history_id=HISTORY_ID, kind="user_prompt", content={"prompt": "test"})
        session.add(test_item)

    with pytest.raises(IntegrityError):
        async with get_test_session() as session:
            await session.delete(history)
