from time import time_ns
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col

from src.history.async_sqlalchemy.adapter import AsyncSqlalchemyHistoryRepo
from src.history.async_sqlalchemy.models import HistoryDb, HistoryItemDb
from src.history.models import UserPrompt
from tests.conftest import get_test_session
from tests.history.utils import compare_user_prompt

HISTORY_ID = uuid4()


async def reset_database():
    async with get_test_session() as session:
        stmt = delete(HistoryItemDb).where(col(HistoryItemDb.history_id) == HISTORY_ID)
        await session.execute(stmt)
        stmt = delete(HistoryDb).where(col(HistoryDb.id) == HISTORY_ID)
        await session.execute(stmt)


@pytest.fixture
def history_repo(engine: AsyncEngine):
    yield AsyncSqlalchemyHistoryRepo(engine)


async def test_history_repo_ops(history_repo: AsyncSqlalchemyHistoryRepo):
    # Setup
    await reset_database()

    # Creating a new history
    history = await history_repo.get_or_create_history(HISTORY_ID)
    assert history.id == HISTORY_ID
    assert history.created_at is not None
    assert history.items == []

    # Adding a history item
    test_user_prompt = UserPrompt(
        id=uuid4(),
        history_id=HISTORY_ID,
        created_at=time_ns(),
        prompt="test prompt",
    )
    await history_repo.add_history_item(test_user_prompt)

    # Getting the history
    history = await history_repo.get_or_create_history(HISTORY_ID)
    assert history.id == HISTORY_ID
    assert history.created_at is not None
    assert len(history.items) == 1
    assert isinstance(history.items[0], UserPrompt)
    compare_user_prompt(history.items[0], test_user_prompt)

    # Teardown
    await reset_database()
