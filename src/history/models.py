from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID


class HistoryItemKind(Enum):
    USER_PROMPT = "user_prompt"
    MODEL_RESPONSE = "model_response"
    THINKING_STEP = "thinking_step"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True)
class BaseHistoryItem:
    id: UUID
    history_id: UUID
    created_at: int


@dataclass(frozen=True)
class UserPrompt(BaseHistoryItem):
    prompt: str


@dataclass(frozen=True)
class ModelResponse(BaseHistoryItem):
    response: str


@dataclass(frozen=True)
class ThinkingStep(BaseHistoryItem):
    thoughts: str


@dataclass(frozen=True)
class ToolCall(BaseHistoryItem):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] | str | None


@dataclass(frozen=True)
class ToolResult(BaseHistoryItem):
    tool_call_id: str
    tool_name: str
    is_retry: bool
    result: Any


HistoryItem = UserPrompt | ModelResponse | ThinkingStep | ToolCall | ToolResult


@dataclass(frozen=True)
class History:
    id: UUID
    created_at: int
    items: list[HistoryItem]
