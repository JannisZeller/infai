from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KeyValueEntry:
    key: str
    collection: str
    value: dict[str, Any]
    expires_at: int | None
