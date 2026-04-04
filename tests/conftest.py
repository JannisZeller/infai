from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.database import DEFAULT_DATABASE_URL, SessionContext, get_engine, get_session, run_migrations
from src.history.repo.port import HistoryRepo

TEST_DATABASE_URL = DEFAULT_DATABASE_URL.replace("database.db", "test.database.db")
TEST_DATABASE_PATH = Path("data/test.database.db")
TEST_DATABASE_PATH.unlink(missing_ok=True)

## Helpers


def as_mock(obj: Any) -> MagicMock:
    return obj


def as_async_mock(obj: Any) -> AsyncMock:
    return obj


#
# Integration Fixtures
#


@asynccontextmanager
async def get_test_session() -> SessionContext:
    run_migrations(TEST_DATABASE_URL)
    engine = get_engine(TEST_DATABASE_URL)

    async with get_session(engine) as session:
        yield session


@pytest.fixture
def engine():
    return get_engine(TEST_DATABASE_URL)


#
# Repository Fixtures
#


@pytest.fixture
def mock_history_repo():
    return AsyncMock(spec=HistoryRepo)
