from src.config.models import Config
from src.core.logging import get_logger
from src.history.service import HistoryService
from src.rag.port import RAGService
from src.rag.qdrant.clients import get_openai_client, get_qdrant_client
from src.rag.qdrant.service import QdrantRAGService

logger = get_logger(__name__, output="console")


async def get_rag_service_or_none(config: Config, history_service: HistoryService) -> RAGService | None:
    if not config.qdrant_url:
        logger.warning("Qdrant URL is not set, running without RAG-memory.")
        return None

    qdrant_client = await get_qdrant_client(config.qdrant_url)
    openai_client = await get_openai_client(config.embedder_config)

    return await QdrantRAGService.create(
        config=config.embedder_config,
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        history_service=history_service,
        history_id=config.history_id,
    )
