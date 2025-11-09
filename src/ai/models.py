from time import time_ns
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.history.models import HistoryItem

LiveItemType = Literal["system_prompt", "model_response_delta", "thinking_delta", "stream_end"]


class LiveItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: int = Field(default_factory=time_ns)


class SystemPrompt(LiveItem):
    content: str


class ModelResponseDelta(LiveItem):
    type_: Literal["model_response_delta"] = "model_response_delta"
    delta: str


class ThinkingDelta(LiveItem):
    type_: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class StreamEnd(LiveItem):
    type_: Literal["stream_end"] = "stream_end"


StreamItem = LiveItem | HistoryItem
