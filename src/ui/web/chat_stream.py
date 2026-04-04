import json
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from src.ai.models import StreamItem

ITEM_TYPE_BY_CLASS_NAME = {
    "SystemPrompt": "system_prompt",
    "PartStart": "part_start",
    "ModelResponseDelta": "model_response_delta",
    "ThinkingDelta": "thinking_delta",
    "StreamEnd": "stream_end",
    "ToolApprovalRequest": "tool_approval_request",
    "MCPAppLaunchRequest": "mcp_app_launch_request",
    "UserPrompt": "user_prompt",
    "ModelResponse": "model_response",
    "ThinkingStep": "thinking_step",
    "ToolCall": "tool_call",
    "ToolResult": "tool_result",
}


async def serialize_stream(stream: AsyncIterator[StreamItem]) -> AsyncIterator[bytes]:
    async for item in stream:
        payload = _to_jsonable(item)
        payload["type"] = _item_type(item)
        yield (json.dumps(payload) + "\n").encode("utf-8")


def _item_type(item: Any) -> str:
    name = item.__class__.__name__
    if name in ITEM_TYPE_BY_CLASS_NAME:
        return ITEM_TYPE_BY_CLASS_NAME[name]
    chars: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value
