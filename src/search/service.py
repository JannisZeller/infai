"""Azure AI Search service for connecting to and querying Azure AI Search resources."""

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SearchResult:
    id: str


class AzureAISearchService:
    """Service class for interacting with Azure AI Search."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        idx: Optional[str] = None,
    ):
        """
        Initialize Azure AI Search service.

        Args:
            url: Azure AI Search endpoint URL. If not provided, uses AZURE_AIS_URL env var.
            key: Azure AI Search API key. If not provided, uses AZURE_AIS_KEY env var.
            idx: Default index name. If not provided, uses AZURE_AIS_IDX env var.
        """
        url = url or os.getenv("AZURE_AIS_URL")
        key = key or os.getenv("AZURE_AIS_KEY")
        idx = idx or os.getenv("AZURE_AIS_IDX")

        if not url:
            raise ValueError("Azure AI Search URL (url) is required")
        if not key:
            raise ValueError("Azure AI Search API (key) is required")
        if not idx:
            raise ValueError("Azure AI Search index (idx) is required")

        self._endpoint = url
        self._index_name = idx
        self._credential = AzureKeyCredential(key)
        self._search_client: Optional[SearchClient] = None
        self._index_client: Optional[SearchIndexClient] = None

    @property
    def search_client(self) -> SearchClient:
        """Get or create SearchClient instance."""
        if self._search_client is None:
            if not self._index_name:
                raise ValueError("Index name is required for SearchClient")
            self._search_client = SearchClient(
                endpoint=self._endpoint,
                index_name=self._index_name,
                credential=self._credential,
            )
        return self._search_client

    @property
    def index_client(self) -> SearchIndexClient:
        """Get or create SearchIndexClient instance."""
        if self._index_client is None:
            self._index_client = SearchIndexClient(
                endpoint=self._endpoint,
                credential=self._credential,
            )
        return self._index_client

    async def search(
        self,
        search_text: str,
        index_name: Optional[str] = None,
        top: int = 5,
        select: Optional[list[str]] = None,
        filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Perform a search query.

        Args:
            search_text: The search query text.
            index_name: Optional index name. Uses default if not provided.
            top: Number of results to return.
            select: List of fields to select in results.
            filter: OData filter expression.

        Returns:
            Search results iterator.
        """
        client = self.search_client
        if index_name and index_name != self._index_name:
            # Create a temporary client for different index
            client = SearchClient(
                endpoint=self._endpoint,
                index_name=index_name,
                credential=self._credential,
            )

        paged_raw_results = client.search(  # type: ignore
            search_text=search_text,
            top=top,
            select=select,
            filter=filter,
        )

        results = []
        raw_results = []

        for raw_result in paged_raw_results:  # type: ignore
            raw_results.append(raw_result)  # type: ignore
            results.append(SearchResult(id=raw_result["id"]))  # type: ignore

        print(raw_results)  # type: ignore

        return results  # type: ignore


if __name__ == "__main__":
    ai_search_service = AzureAISearchService(idx="test")
    results = asyncio.run(ai_search_service.search(search_text="*"))
    print(results)
