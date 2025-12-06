from dataclasses import dataclass
from uuid import UUID

from src.history.service.models import HistoryItem


@dataclass(frozen=True)
class LiveItem:
    id: UUID
    created_at: int


@dataclass(frozen=True)
class SystemPrompt(LiveItem):
    prompt: str


@dataclass(frozen=True)
class ModelResponseDelta(LiveItem):
    delta: str


@dataclass(frozen=True)
class ThinkingDelta(LiveItem):
    delta: str


@dataclass(frozen=True)
class StreamEnd(LiveItem):
    pass


StreamItem = LiveItem | HistoryItem
