from functools import lru_cache

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from src.config.models import EmbedderConfig
from src.core.exceptions import ResourceNotAvailableError
from src.core.logging import get_logger

logger = get_logger("Startup: ", output="console", simple_format=True)


@lru_cache
async def get_qdrant_client(qdrant_url: str) -> AsyncQdrantClient:
    client = AsyncQdrantClient(url=qdrant_url)

    logger.info(f"Qdrant: Pinging {qdrant_url}...")
    try:
        await client.get_collections()
    except Exception as e:
        logger.error(f"Qdrant: Pinging {qdrant_url} failed with error: {e}")
        raise ResourceNotAvailableError(qdrant_url)

    return client


@lru_cache
async def get_openai_client(cfg: EmbedderConfig) -> AsyncOpenAI:
    client = AsyncOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
    )

    logger.info(f"Embedder: Pinging {cfg.model_name}...")
    try:
        await client.embeddings.create(
            input="test",
            model=cfg.model_name,
        )
    except Exception as e:
        raise ResourceNotAvailableError(f"Embedder: Pinging {cfg.model_name} failed with error: {e}")

    return client
