from typing import Protocol
from uuid import UUID

from src.history.models import History, HistoryItem


class HistoryRepo(Protocol):
    """The Port that the HistoryService expects."""

    async def get_or_create_history(self, history_id: UUID) -> History: ...

    async def add_history_item(self, history_item: HistoryItem) -> None: ...
