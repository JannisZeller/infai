from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from src.history.models import HistoryItem


@dataclass(frozen=True)
class BaseLiveItem:
    id: UUID
    history_id: UUID
    created_at: int


@dataclass(frozen=True)
class SystemPrompt(BaseLiveItem):
    prompt: str


PartType = Literal["thinking", "response", "tool_call_prep", "final_response"]


@dataclass(frozen=True)
class PartStart(BaseLiveItem):
    part_type: PartType


@dataclass(frozen=True)
class ModelResponseDelta(BaseLiveItem):
    delta: str


@dataclass(frozen=True)
class ThinkingDelta(BaseLiveItem):
    delta: str


@dataclass(frozen=True)
class StreamEnd(BaseLiveItem):
    pass


LiveItem = SystemPrompt | PartStart | ModelResponseDelta | ThinkingDelta | StreamEnd

StreamItem = LiveItem | HistoryItem
