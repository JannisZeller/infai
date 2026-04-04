from collections.abc import Mapping, Sequence
from time import time_ns
from typing import Any, SupportsFloat, cast

from sqlalchemy import delete, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlmodel import col

from src.core.database import get_session
from src.token_store.async_sqlalchemy.mapper import TokenStoreCipher
from src.token_store.async_sqlalchemy.models import TokenStoreEntryDb
from src.token_store.port import TokenStore

NANOSECONDS_PER_SECOND = 1_000_000_000


class AsyncSqlalchemyTokenStore(TokenStore):
    def __init__(
        self,
        engine: AsyncEngine,
        encryption_key: str,
        default_collection: str = "default",
    ):
        if not default_collection:
            raise ValueError("default_collection must not be empty.")

        self._engine = engine
        self._default_collection = default_collection
        self._cipher = TokenStoreCipher(encryption_key=encryption_key)

    async def get(self, key: str, *, collection: str | None = None) -> dict[str, Any] | None:
        value, _ = await self.ttl(key=key, collection=collection)
        return value

    async def ttl(self, key: str, *, collection: str | None = None) -> tuple[dict[str, Any] | None, float | None]:
        values = await self.ttl_many(keys=[key], collection=collection)
        if not values:
            return (None, None)
        return values[0]

    async def put(
        self,
        key: str,
        value: Mapping[str, Any],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        await self.put_many(keys=[key], values=[value], collection=collection, ttl=ttl)

    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        return (await self.delete_many(keys=[key], collection=collection)) > 0

    async def get_many(self, keys: Sequence[str], *, collection: str | None = None) -> list[dict[str, Any] | None]:
        return [value for value, _ in await self.ttl_many(keys=keys, collection=collection)]

    async def ttl_many(
        self,
        keys: Sequence[str],
        *,
        collection: str | None = None,
    ) -> list[tuple[dict[str, Any] | None, float | None]]:
        if not keys:
            return []

        resolved_collection = self._resolve_collection(collection)
        unique_keys = list(dict.fromkeys(keys))
        now = time_ns()

        async with get_session(self._engine) as session:
            entries_by_key = await self._select_entries_by_key(
                session=session,
                keys=unique_keys,
                collection=resolved_collection,
            )

            expired_keys = [
                key
                for key, entry in entries_by_key.items()
                if self._is_expired(entry_expires_at=entry.expires_at, now=now)
            ]
            if expired_keys:
                await session.execute(
                    delete(TokenStoreEntryDb).where(
                        col(TokenStoreEntryDb.collection) == resolved_collection,
                        col(TokenStoreEntryDb.key).in_(expired_keys),
                    )
                )
                for expired_key in expired_keys:
                    entries_by_key.pop(expired_key, None)

            return [self._to_ttl_result(entry=entries_by_key.get(key), now=now) for key in keys]

    async def put_many(
        self,
        keys: Sequence[str],
        values: Sequence[Mapping[str, Any]],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length.")
        if not keys:
            return

        resolved_collection = self._resolve_collection(collection)
        unique_keys = list(dict.fromkeys(keys))
        now = time_ns()
        expires_at = self._to_expires_at(now=now, ttl=ttl)

        async with get_session(self._engine) as session:
            entries_by_key = await self._select_entries_by_key(
                session=session,
                keys=unique_keys,
                collection=resolved_collection,
            )

            for key, value in zip(keys, values, strict=True):
                ciphertext = self._cipher.encrypt(value)
                existing = entries_by_key.get(key)
                if existing:
                    existing.ciphertext = ciphertext
                    existing.expires_at = expires_at
                    existing.updated_at = now
                    continue

                entry = TokenStoreEntryDb(
                    collection=resolved_collection,
                    key=key,
                    ciphertext=ciphertext,
                    expires_at=expires_at,
                    created_at=now,
                    updated_at=now,
                )
                entries_by_key[key] = entry
                session.add(entry)

    async def delete_many(self, keys: Sequence[str], *, collection: str | None = None) -> int:
        if not keys:
            return 0

        resolved_collection = self._resolve_collection(collection)

        async with get_session(self._engine) as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    delete(TokenStoreEntryDb).where(
                        col(TokenStoreEntryDb.collection) == resolved_collection,
                        col(TokenStoreEntryDb.key).in_(list(dict.fromkeys(keys))),
                    )
                ),
            )

            return max(result.rowcount or 0, 0)

    async def cull_expired(self) -> int:
        now = time_ns()
        async with get_session(self._engine) as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    delete(TokenStoreEntryDb).where(
                        col(TokenStoreEntryDb.expires_at).is_not(None),
                        col(TokenStoreEntryDb.expires_at) <= now,
                    )
                ),
            )

            return max(result.rowcount or 0, 0)

    async def _select_entries_by_key(
        self,
        session: AsyncSession,
        keys: Sequence[str],
        collection: str,
    ) -> dict[str, TokenStoreEntryDb]:
        if not keys:
            return {}

        query = select(TokenStoreEntryDb).where(
            col(TokenStoreEntryDb.collection) == collection,
            col(TokenStoreEntryDb.key).in_(keys),
        )
        result = await session.execute(query)
        entries = result.scalars().all()
        return {entry.key: entry for entry in entries}

    def _to_ttl_result(
        self,
        entry: TokenStoreEntryDb | None,
        now: int,
    ) -> tuple[dict[str, Any] | None, float | None]:
        if entry is None:
            return (None, None)

        value = self._cipher.decrypt(entry.ciphertext)
        if entry.expires_at is None:
            return (value, None)

        ttl_seconds = max(entry.expires_at - now, 0) / NANOSECONDS_PER_SECOND
        return (value, ttl_seconds)

    def _resolve_collection(self, collection: str | None) -> str:
        resolved = collection or self._default_collection
        if not resolved:
            raise ValueError("collection must not be empty.")
        return resolved

    def _is_expired(self, entry_expires_at: int | None, now: int) -> bool:
        return entry_expires_at is not None and entry_expires_at <= now

    def _to_expires_at(self, now: int, ttl: SupportsFloat | None) -> int | None:
        if ttl is None:
            return None

        ttl_seconds = float(ttl)
        return now + int(ttl_seconds * NANOSECONDS_PER_SECOND)
