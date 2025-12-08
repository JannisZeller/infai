import os
from uuid import UUID

from openai import AsyncAzureOpenAI
from qdrant_client import AsyncQdrantClient


class QdrantClientProvider:
    def __init__(self, qdrant_url: str):
        self._qdrant_client = AsyncQdrantClient(url=qdrant_url)

    def get_client(self) -> AsyncQdrantClient:
        return self._qdrant_client


class AzureOpenAIClientProvider:
    def __init__(self, history_id: UUID):
        self._azure_openai_client = AsyncAzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_AIF_URL") or "",
            api_key=os.getenv("AZURE_AIF_KEY") or "",
        )

    def get_client(self) -> AsyncAzureOpenAI:
        return self._azure_openai_client
