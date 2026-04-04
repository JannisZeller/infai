from collections.abc import Mapping
from typing import cast
from uuid import uuid4

from cryptography.fernet import Fernet
from fastmcp.client.auth.oauth import OAuthToken, TokenStorageAdapter
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col

from src.token_store.async_sqlalchemy.adapter import AsyncSqlalchemyTokenStore
from src.token_store.async_sqlalchemy.models import TokenStoreEntryDb
from tests.conftest import get_test_session

TEST_COLLECTION = "token-store-test"


async def reset_token_store() -> None:
    async with get_test_session() as session:
        stmt = delete(TokenStoreEntryDb).where(col(TokenStoreEntryDb.collection) == TEST_COLLECTION)
        await session.execute(stmt)


def _get_store(engine: AsyncEngine) -> AsyncSqlalchemyTokenStore:
    return AsyncSqlalchemyTokenStore(
        engine=engine,
        encryption_key=Fernet.generate_key().decode("utf-8"),
        default_collection=TEST_COLLECTION,
    )


async def test_token_store_put_get_roundtrip_is_encrypted(engine: AsyncEngine):
    await reset_token_store()
    store = _get_store(engine)
    key = f"server-{uuid4()}/tokens"
    value: Mapping[str, str] = {
        "access_token": "super-secret-token",
        "token_type": "bearer",
    }

    await store.put(key=key, value=value)

    async_key_value_store = cast(object, store)
    assert async_key_value_store is not None
    assert await store.get(key=key) == value

    async with get_test_session() as session:
        stmt = select(TokenStoreEntryDb).where(
            col(TokenStoreEntryDb.collection) == TEST_COLLECTION,
            col(TokenStoreEntryDb.key) == key,
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        ciphertext = row.ciphertext if row is not None else None

    assert row is not None
    assert ciphertext is not None
    assert b"super-secret-token" not in ciphertext

    assert await store.delete(key=key)
    assert await store.get(key=key) is None


async def test_token_store_ttl_and_bulk_ops(engine: AsyncEngine):
    await reset_token_store()
    store = _get_store(engine)

    await store.put_many(
        keys=["k1", "k2"],
        values=[{"value": "v1"}, {"value": "v2"}],
        ttl=60,
    )

    assert await store.get_many(keys=["k1", "k2", "missing"]) == [{"value": "v1"}, {"value": "v2"}, None]

    k1_value, k1_ttl = await store.ttl(key="k1")
    assert k1_value == {"value": "v1"}
    assert k1_ttl is not None
    assert k1_ttl > 0

    deleted = await store.delete_many(keys=["k1", "missing"])
    assert deleted == 1

    await store.put(key="expiring", value={"value": "soon-gone"}, ttl=0)
    assert await store.get(key="expiring") is None

    await reset_token_store()


async def test_token_store_is_fastmcp_oauth_compatible(engine: AsyncEngine):
    await reset_token_store()
    store = _get_store(engine)
    token_storage_adapter = TokenStorageAdapter(async_key_value=store, server_url="https://example.com/mcp")

    token = OAuthToken(
        access_token="access-token-value",
        token_type="Bearer",
        expires_in=3600,
        scope="read",
        refresh_token="refresh-token-value",
    )

    await token_storage_adapter.set_tokens(token)
    stored_token = await token_storage_adapter.get_tokens()

    assert stored_token is not None
    assert stored_token.model_dump() == token.model_dump()
