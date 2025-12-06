from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class HistoryItem:
    id: UUID
    history_id: UUID
    created_at: int


@dataclass(frozen=True)
class UserPrompt(HistoryItem):
    prompt: str


@dataclass(frozen=True)
class ModelResponse(HistoryItem):
    response: str


@dataclass(frozen=True)
class ThinkingStep(HistoryItem):
    thoughts: str


@dataclass(frozen=True)
class ToolCall(HistoryItem):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] | str | None


@dataclass(frozen=True)
class ToolResult(HistoryItem):
    tool_call_id: str
    tool_name: str
    is_retry: bool
    result: Any


@dataclass(frozen=True)
class History:
    id: UUID
    created_at: int
    items: list[HistoryItem]
