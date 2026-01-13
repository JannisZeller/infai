from uuid import UUID

from pydantic import BaseModel

from src.history.models import HistoryItemKind


class QdrantRAGItem(BaseModel):
    history_item_id: UUID
    history_id: UUID
    created_at: int
    text: str
    kind: HistoryItemKind


Embedding = list[float]
