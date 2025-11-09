from time import time_ns
from typing import Annotated, Any, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Discriminator, Field

HistoryItemType = Literal[
    "user_prompt",
    "model_response",
    "thinking_step",
    "tool_call",
    "tool_result",
]


class _BaseHistoryItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: int = Field(default_factory=time_ns)


class UserPrompt(_BaseHistoryItem):
    type_: Literal["user_prompt"] = "user_prompt"
    prompt: str


class ModelResponse(_BaseHistoryItem):
    type_: Literal["model_response"] = "model_response"
    response: str


class ThinkingStep(_BaseHistoryItem):
    type_: Literal["thinking_step"] = "thinking_step"
    thoughts: str


class ToolCall(_BaseHistoryItem):
    type_: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] | str | None


class ToolResult(_BaseHistoryItem):
    type_: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    is_retry: bool
    result: Any


HistoryItem = Annotated[Union[UserPrompt, ModelResponse, ThinkingStep, ToolCall, ToolResult], Discriminator("type_")]


class History(BaseModel):
    items: list[HistoryItem]
