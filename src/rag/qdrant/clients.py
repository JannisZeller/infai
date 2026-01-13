import os
from uuid import UUID

from openai import AsyncAzureOpenAI, AsyncOpenAI
from qdrant_client import AsyncQdrantClient


class QdrantClientProvider:
    def __init__(self, qdrant_url: str):
        self._qdrant_client = AsyncQdrantClient(url=qdrant_url)

    def get_client(self) -> AsyncQdrantClient:
        return self._qdrant_client


class AzureOpenAIClientProvider:
    def __init__(self, history_id: UUID):
        api_key = os.getenv("AZURE_AIF_KEY")
        if not api_key:
            raise ValueError("AZURE_AIF_KEY is not set in .env")
        base_url = os.getenv("AZURE_AIF_URL")
        if not base_url:
            raise ValueError("AZURE_AIF_URL is not set in .env")
        self._azure_openai_client = AsyncAzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=base_url,
            api_key=api_key,
        )

    def get_client(self) -> AsyncAzureOpenAI:
        return self._azure_openai_client


class OpenAIProvider:
    def __init__(self, history_id: UUID):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("OPENAI_BASE_URL is not set in .env")
        self._openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def get_client(self) -> AsyncOpenAI:
        return self._openai_client
