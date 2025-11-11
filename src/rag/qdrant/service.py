from textwrap import dedent
from time import time_ns
from uuid import UUID, uuid4

from openai import AsyncAzureOpenAI, AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qdm

from src.ai.models import SystemPrompt
from src.config.models import EmbedderConfig
from src.history.models import HistoryItem, ModelResponse, UserPrompt
from src.history.service import HistoryService
from src.rag.qdrant.mapper import QdrantRAGMapper
from src.rag.qdrant.models import Embedding, QdrantRAGItem


class QdrantRAGService:
    _qdrant_client: AsyncQdrantClient
    _openai_client: AsyncAzureOpenAI | AsyncOpenAI
    _history_service: HistoryService
    _history_id: UUID
    _collection_name: str
    _embedding_model_name: str
    _embedding_dimensions: int
    _embedding_chunk_max_chars: int
    _embedding_chunk_overlap_chars: int

    @classmethod
    async def create(
        cls,
        config: EmbedderConfig,
        qdrant_client: AsyncQdrantClient,
        openai_client: AsyncOpenAI,
        history_service: HistoryService,
        history_id: UUID,
    ):
        self = cls()
        self._qdrant_client = qdrant_client
        self._openai_client = openai_client
        self._history_service = history_service
        self._history_id = history_id
        self._collection_name = f"history-{history_id}"
        self._embedding_model_name = config.model_name
        self._embedding_chunk_max_chars = config.chunk_max_chars
        self._embedding_chunk_overlap_chars = config.chunk_overlap_chars

        # Setting up the embedding model
        await self._store_embedding_dimensions()
        await self._create_collection_if_not_exists()
        return self

    async def _create_collection_if_not_exists(self):
        if not self._embedding_dimensions:
            raise ValueError("Embedding dimensions are not set")
        if not self._collection_name:
            raise ValueError("Collection name is not set")
        if not await self._qdrant_client.collection_exists(self._collection_name):
            await self._qdrant_client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qdm.VectorParams(size=self._embedding_dimensions, distance=qdm.Distance.COSINE),
            )

    async def _store_embedding_dimensions(self):
        sample_embedding = await self._openai_client.embeddings.create(
            input="test",
            model=self._embedding_model_name,
        )
        self._embedding_dimensions = len(sample_embedding.data[0].embedding)

    def _chunk_rag_doc(self, rag_doc: QdrantRAGItem) -> list[QdrantRAGItem]:
        text = rag_doc.text
        chunked_rag_docs: list[QdrantRAGItem] = []

        if len(text) <= self._embedding_chunk_max_chars:
            return [rag_doc]

        def _create_rag_item(text: str) -> QdrantRAGItem:
            return QdrantRAGItem(
                history_item_id=rag_doc.history_item_id,
                history_id=rag_doc.history_id,
                created_at=rag_doc.created_at,
                text=text,
                kind=rag_doc.kind,
            )

        while len(text) > self._embedding_chunk_max_chars:
            chunked_rag_docs.append(_create_rag_item(text[: self._embedding_chunk_max_chars]))
            text = text[self._embedding_chunk_max_chars - self._embedding_chunk_overlap_chars :]

        return chunked_rag_docs

    def _chunk_rag_docs(self, rag_docs: list[QdrantRAGItem]) -> list[QdrantRAGItem]:
        chunked_rag_docs: list[QdrantRAGItem] = []
        for rag_doc in rag_docs:
            chunked_rag_docs.extend(self._chunk_rag_doc(rag_doc))
        return chunked_rag_docs

    async def _embed_rag_docs(self, rag_docs: list[QdrantRAGItem]) -> list[Embedding]:
        ebd_results = await self._openai_client.embeddings.create(
            input=[rag_doc.text for rag_doc in rag_docs],
            model=self._embedding_model_name,
        )
        return [ebd_result.embedding for ebd_result in ebd_results.data]

    async def _upsert_rag_docs_and_embeddings(self, rag_docs: list[QdrantRAGItem], embeddings: list[Embedding]):
        for rag_doc, embedding in zip(rag_docs, embeddings):
            await self._qdrant_client.upsert(
                collection_name=self._collection_name,
                points=[
                    qdm.PointStruct(
                        id=str(uuid4()),
                        vector=embedding,
                        payload=rag_doc.model_dump(mode="json"),
                    )
                ],
            )

    async def add_history_items(self, history_items: list[UserPrompt | ModelResponse]):
        rag_docs = QdrantRAGMapper.map_history_items_to_rag_items(history_items)
        chunked_rag_docs = self._chunk_rag_docs(rag_docs)
        embeddings = await self._embed_rag_docs(chunked_rag_docs)
        await self._upsert_rag_docs_and_embeddings(chunked_rag_docs, embeddings)

    async def _search_for_embedding(self, embedding: Embedding, top_k: int) -> list[HistoryItem]:
        results = await self._qdrant_client.query_points(
            collection_name=self._collection_name,
            query_filter=qdm.Filter(
                must=[
                    qdm.FieldCondition(
                        key="history_id",
                        match=qdm.MatchValue(value=str(self._history_id)),
                    )
                ]
            ),
            query=embedding,
            limit=top_k,
        )
        points = results.points

        history_items: list[HistoryItem] = []
        for point in points:
            rag_item = QdrantRAGItem.model_validate(point.payload)
            history_item = QdrantRAGMapper.map_point_to_history_item(rag_item)
            history_items.append(history_item)

        return history_items

    async def search_for_user_prompt(self, user_prompt: UserPrompt, top_k: int = 10) -> SystemPrompt:
        rag_doc = QdrantRAGMapper.map_history_item_to_rag_item(user_prompt)
        max_len_search_rag_doc = self._chunk_rag_docs([rag_doc])[0]
        embeddings = await self._embed_rag_docs([max_len_search_rag_doc])
        history_items = await self._search_for_embedding(embeddings[0], top_k)
        if len(history_items) == 0:
            return SystemPrompt(
                id=uuid4(),
                history_id=user_prompt.history_id,
                created_at=time_ns(),
                prompt=dedent("""
                    [# Relevant Previous Interactions #]

                    No relevant previous interactions between the user and you (the assistant) have been found.
                """),
            )
        # TODO: Add "days ago"
        prompt = dedent("""
            [# Relevant Previous Interactions #]

            Via a semantic search, the following previous messages between the user and you (the assistant) have been found to be relevant to the current user prompt.

            <previous_interactions>

        """).strip()

        # TODO: Improve formatting
        for history_item in history_items:
            if isinstance(history_item, UserPrompt):
                prompt += f"\n\t<user_prompt>\n\t{history_item.prompt}\n\t</user_prompt>\n"
            elif isinstance(history_item, ModelResponse):
                prompt += f"\n\t<model_response>\n\t{history_item.response}\n\t</model_response>\n"
            else:
                raise NotImplementedError(f"Unexpected history item: {history_item} to construct RAG system prompt")

        prompt += "\n\n</previous_interactions>"
        return SystemPrompt(
            id=uuid4(),
            history_id=user_prompt.history_id,
            created_at=time_ns(),
            prompt=prompt,
        )

    # TODO: Tests
    # TODO: Long term explicit memory via tool
