from unittest.mock import AsyncMock

from src.token_store.session_scoped import SessionScopedAsyncKeyValue


async def test_session_scoped_async_key_value_prefixes_keys():
    store = AsyncMock()
    scoped_store = SessionScopedAsyncKeyValue(key_value=store, session_id="session-123")

    await scoped_store.put("token-key", {"access_token": "value"}, collection="tokens")
    await scoped_store.get("token-key", collection="tokens")

    store.put.assert_awaited_once_with(
        "session-123:token-key", {"access_token": "value"}, collection="tokens", ttl=None
    )
    store.get.assert_awaited_once_with("session-123:token-key", collection="tokens")
