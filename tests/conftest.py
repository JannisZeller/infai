from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.database import SessionContext, create_db_and_tables, get_engine, get_session
from src.history.port import HistoryRepo


def as_mock(obj: Any) -> MagicMock:
    return obj


@asynccontextmanager
async def get_test_session() -> SessionContext:
    engine = get_engine()
    await create_db_and_tables(engine)

    async with get_session(engine) as session:
        yield session


@pytest.fixture
def engine():
    return get_engine()


@pytest.fixture
def mock_history_repo():
    return AsyncMock(spec=HistoryRepo)
