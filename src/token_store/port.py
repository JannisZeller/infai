from typing import Protocol

from key_value.aio.protocols import AsyncKeyValue


class TokenStore(AsyncKeyValue, Protocol):
    async def cull_expired(self) -> int: ...
