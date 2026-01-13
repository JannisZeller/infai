from typing import Sequence
from uuid import UUID

from src.ai.models import SystemPrompt
from src.history.models import History, HistoryItem
from src.history.port import HistoryRepo


class HistoryService:
    def __init__(self, history_repo: HistoryRepo):
        self._history_repo = history_repo

    async def get_or_create_history_by_id(self, history_id: UUID) -> History:
        return await self._history_repo.get_or_create_history(history_id)

    async def add_history_item(self, history_item: HistoryItem):
        await self._history_repo.add_history_item(history_item)

    async def get_last_n_history_items(self, history_id: UUID, n: int) -> Sequence[HistoryItem | SystemPrompt]:
        history = await self.get_or_create_history_by_id(history_id)
        return history.items[-n:]
