from time import time_ns
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import joinedload
from sqlmodel import col

from src.core.database import get_session
from src.history.async_sqlalchemy.mapper import map_history_item_to_db, map_history_item_to_domain
from src.history.async_sqlalchemy.models import HistoryDb
from src.history.models import History, HistoryItem


class AsyncSqlalchemyHistoryRepo:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def _find_history_db_by_id_eager(self, history_id: UUID) -> History | None:
        async with get_session(self._engine) as session:
            query = select(HistoryDb).where(col(HistoryDb.id) == history_id).options(joinedload(HistoryDb.items))  # type: ignore
            result = await session.execute(query)
            history_db = result.unique().scalar_one_or_none()
            if not history_db:
                return None
            return History(
                id=history_db.id,
                created_at=history_db.created_at,
                items=[map_history_item_to_domain(item) for item in history_db.items],
            )

    async def _add_history(self, history_db: HistoryDb):
        async with get_session(self._engine) as session:
            session.add(history_db)

    async def get_or_create_history(self, history_id: UUID) -> History:
        history = await self._find_history_db_by_id_eager(history_id)
        if not history:
            history_db = HistoryDb(id=history_id, created_at=time_ns(), items=[])
            history = History(
                id=history_db.id,
                created_at=history_db.created_at,
                items=[map_history_item_to_domain(item) for item in history_db.items],
            )
            await self._add_history(history_db)
        return history

    async def add_history_item(self, history_item: HistoryItem):
        async with get_session(self._engine) as session:
            history_item_db = map_history_item_to_db(history_item)
            session.add(history_item_db)
