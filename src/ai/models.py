from dataclasses import dataclass
from typing import Any, Literal
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


@dataclass(frozen=True)
class ToolApprovalRequest(BaseLiveItem):
    resume_token: str
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] | str | None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolApprovalDecision:
    tool_call_id: str
    approved: bool
    denial_message: str | None = None


@dataclass(frozen=True)
class MCPAppLaunchRequest(BaseLiveItem):
    tool_call_id: str
    tool_name: str
    url: str
    title: str | None = None
    source: Literal["tool_result", "tool_result_metadata"] = "tool_result"


LiveItem = (
    SystemPrompt
    | PartStart
    | ModelResponseDelta
    | ThinkingDelta
    | StreamEnd
    | ToolApprovalRequest
    | MCPAppLaunchRequest
)

StreamItem = LiveItem | HistoryItem
