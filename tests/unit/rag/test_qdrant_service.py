from time import time_ns
from uuid import uuid4

from src.history.models import HistoryItemKind
from src.rag.qdrant.models import QdrantRAGItem
from src.rag.qdrant.service import QdrantRAGService


def test_chunk_rag_doc_keeps_trailing_chunk():
    service = QdrantRAGService()
    service._embedding_chunk_max_chars = 5
    service._embedding_chunk_overlap_chars = 2

    rag_doc = QdrantRAGItem(
        history_item_id=uuid4(),
        history_id=uuid4(),
        created_at=time_ns(),
        text="abcdefghij",
        kind=HistoryItemKind.USER_PROMPT,
    )

    chunked_docs = service._chunk_rag_doc(rag_doc)

    assert [chunk.text for chunk in chunked_docs] == ["abcde", "defgh", "ghij"]
