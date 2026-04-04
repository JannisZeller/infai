from key_value.aio.protocols import AsyncKeyValue
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config.models import Config
from src.token_store.async_sqlalchemy.adapter import AsyncSqlalchemyTokenStore


def get_token_store_or_none(config: Config, engine: AsyncEngine) -> AsyncKeyValue | None:
    if not config.token_store:
        return None

    return AsyncSqlalchemyTokenStore(
        engine=engine,
        encryption_key=config.token_store.encryption_key,
        default_collection=config.token_store.default_collection,
    )
