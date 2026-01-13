from time import time_ns
from uuid import uuid4

import pytest

from src.history.models import History, UserPrompt
from src.history.port import HistoryRepo
from src.history.service import HistoryService
from tests.conftest import as_mock
from tests.history.utils import compare_user_prompt

HISTORY_ID = uuid4()


@pytest.fixture
def history_service(mock_history_repo: HistoryRepo):
    return HistoryService(history_repo=mock_history_repo)


async def test_get_or_create_history_by_id(
    history_service: HistoryService,
    mock_history_repo: HistoryRepo,
):
    # Setup
    test_user_prompt = UserPrompt(
        id=uuid4(),
        history_id=HISTORY_ID,
        created_at=time_ns(),
        prompt="test prompt",
    )
    test_history = History(id=HISTORY_ID, created_at=time_ns(), items=[test_user_prompt])

    # Execute
    as_mock(mock_history_repo.get_or_create_history).return_value = test_history
    history = await history_service.get_or_create_history_by_id(HISTORY_ID)

    # Assert
    as_mock(mock_history_repo.get_or_create_history).assert_called_once_with(HISTORY_ID)
    assert history.id == HISTORY_ID
    assert history.created_at is not None
    assert len(history.items) == 1
    assert isinstance(history.items[0], UserPrompt)
    compare_user_prompt(history.items[0], test_user_prompt)


async def test_add_history_item(
    history_service: HistoryService,
    mock_history_repo: HistoryRepo,
):
    # Setup
    test_user_prompt = UserPrompt(
        id=uuid4(),
        history_id=HISTORY_ID,
        created_at=time_ns(),
        prompt="test prompt",
    )

    as_mock(mock_history_repo.add_history_item).return_value = test_user_prompt
    await history_service.add_history_item(test_user_prompt)

    # Assert
    as_mock(mock_history_repo.add_history_item).assert_called_once_with(test_user_prompt)
