from uuid import UUID

from src.history.repo.repo import HistoryRepo
from src.history.service.models import History, HistoryItem


class HistoryService:
    def __init__(self, repo: HistoryRepo):
        self._repo = repo

    async def get_or_create_history_by_id(self, history_id: UUID) -> History:
        return await self._repo.get_or_create_history(history_id)

    async def add_history_item(self, history_item: HistoryItem):
        await self._repo.add_history_item(history_item)
