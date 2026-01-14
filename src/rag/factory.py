from src.config.models import Config
from src.history.service import HistoryService
from src.rag.port import RAGService
from src.rag.qdrant.clients import OpenAIProvider, QdrantClientProvider
from src.rag.qdrant.service import QdrantRAGService


async def get_rag_service(config: Config, history_service: HistoryService) -> RAGService:
    qdrant_client_provider = QdrantClientProvider(qdrant_url=config.qdrant_url)
    openai_client_provider = OpenAIProvider(history_id=config.history_id)

    return await QdrantRAGService.create(
        qdrant_client_provider=qdrant_client_provider,
        openai_client_provider=openai_client_provider,
        history_service=history_service,
        history_id=config.history_id,
    )
