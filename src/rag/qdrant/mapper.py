from src.history.models import HistoryItem, HistoryItemKind, ModelResponse, UserPrompt
from src.rag.qdrant.models import QdrantRAGItem


class QdrantRAGMapper:
    @staticmethod
    def map_history_item_to_rag_item(history_item: UserPrompt | ModelResponse) -> QdrantRAGItem:
        if isinstance(history_item, UserPrompt):
            return QdrantRAGItem(
                history_item_id=history_item.id,
                history_id=history_item.history_id,
                created_at=history_item.created_at,
                text=history_item.prompt,
                kind=HistoryItemKind.USER_PROMPT,
            )
        elif isinstance(history_item, ModelResponse):  # type: ignore - staying explicit
            return QdrantRAGItem(
                history_item_id=history_item.id,
                history_id=history_item.history_id,
                created_at=history_item.created_at,
                text=history_item.response,
                kind=HistoryItemKind.MODEL_RESPONSE,
            )
        else:
            raise NotImplementedError(f"Unexpected history item: {history_item} to map to plain text for embedding")

    @staticmethod
    def map_point_to_history_item(rag_item: QdrantRAGItem) -> HistoryItem:
        if rag_item.kind == HistoryItemKind.USER_PROMPT:
            return UserPrompt(
                id=rag_item.history_item_id,
                history_id=rag_item.history_id,
                created_at=rag_item.created_at,
                prompt=rag_item.text,
            )
        elif rag_item.kind == HistoryItemKind.MODEL_RESPONSE:
            return ModelResponse(
                id=rag_item.history_item_id,
                history_id=rag_item.history_id,
                created_at=rag_item.created_at,
                response=rag_item.text,
            )
        else:
            raise NotImplementedError(f"Unexpected history item: {rag_item} to map to history item")

    @staticmethod
    def map_history_items_to_rag_items(history_items: list[UserPrompt | ModelResponse]) -> list[QdrantRAGItem]:
        return [QdrantRAGMapper.map_history_item_to_rag_item(item) for item in history_items]
