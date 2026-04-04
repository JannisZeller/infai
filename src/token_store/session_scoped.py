from collections.abc import Mapping, Sequence
from typing import Any, SupportsFloat

from key_value.aio.protocols import AsyncKeyValue


class SessionScopedAsyncKeyValue:
    def __init__(self, key_value: AsyncKeyValue, session_id: str):
        self._key_value = key_value
        self._session_id = session_id

    async def get(self, key: str, *, collection: str | None = None) -> dict[str, Any] | None:
        return await self._key_value.get(self._key(key), collection=collection)

    async def ttl(self, key: str, *, collection: str | None = None) -> tuple[dict[str, Any] | None, float | None]:
        return await self._key_value.ttl(self._key(key), collection=collection)

    async def put(
        self,
        key: str,
        value: Mapping[str, Any],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        await self._key_value.put(self._key(key), value, collection=collection, ttl=ttl)

    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        return await self._key_value.delete(self._key(key), collection=collection)

    async def get_many(self, keys: Sequence[str], *, collection: str | None = None) -> list[dict[str, Any] | None]:
        return await self._key_value.get_many([self._key(key) for key in keys], collection=collection)

    async def ttl_many(
        self,
        keys: Sequence[str],
        *,
        collection: str | None = None,
    ) -> list[tuple[dict[str, Any] | None, float | None]]:
        return await self._key_value.ttl_many([self._key(key) for key in keys], collection=collection)

    async def put_many(
        self,
        keys: Sequence[str],
        values: Sequence[Mapping[str, Any]],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        await self._key_value.put_many([self._key(key) for key in keys], values, collection=collection, ttl=ttl)

    async def delete_many(self, keys: Sequence[str], *, collection: str | None = None) -> int:
        return await self._key_value.delete_many([self._key(key) for key in keys], collection=collection)

    def _key(self, key: str) -> str:
        return f"{self._session_id}:{key}"
