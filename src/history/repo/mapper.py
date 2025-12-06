from enum import Enum

from src.history.repo.models import HistoryItemDb
from src.history.service.models import (
    HistoryItem,
    ModelResponse,
    ThinkingStep,
    ToolCall,
    ToolResult,
    UserPrompt,
)


class HistoryItemKind(Enum):
    USER_PROMPT = "user_prompt"
    MODEL_RESPONSE = "model_response"
    THINKING_STEP = "thinking_step"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


def map_history_item_to_db(history_item: HistoryItem) -> HistoryItemDb:
    if isinstance(history_item, UserPrompt):
        return HistoryItemDb(
            history_id=history_item.history_id,
            created_at=history_item.created_at,
            kind=HistoryItemKind.USER_PROMPT.value,
            content={"prompt": history_item.prompt},
        )
    elif isinstance(history_item, ModelResponse):
        return HistoryItemDb(
            history_id=history_item.history_id,
            created_at=history_item.created_at,
            kind=HistoryItemKind.MODEL_RESPONSE.value,
            content={"response": history_item.response},
        )
    elif isinstance(history_item, ThinkingStep):
        return HistoryItemDb(
            history_id=history_item.history_id,
            created_at=history_item.created_at,
            kind=HistoryItemKind.THINKING_STEP.value,
            content={"thoughts": history_item.thoughts},
        )
    elif isinstance(history_item, ToolCall):
        return HistoryItemDb(
            history_id=history_item.history_id,
            created_at=history_item.created_at,
            kind=HistoryItemKind.TOOL_CALL.value,
            content={
                "tool_call_id": history_item.tool_call_id,
                "tool_name": history_item.tool_name,
                "args": history_item.args,
            },
        )
    elif isinstance(history_item, ToolResult):
        return HistoryItemDb(
            history_id=history_item.history_id,
            created_at=history_item.created_at,
            kind=HistoryItemKind.TOOL_RESULT.value,
            content={
                "tool_call_id": history_item.tool_call_id,
                "tool_name": history_item.tool_name,
                "is_retry": history_item.is_retry,
                "result": history_item.result,
            },
        )
    else:
        raise ValueError(f"Unexpected history item: {history_item}")


def map_history_item_to_domain(history_item_db: HistoryItemDb) -> HistoryItem:
    if history_item_db.kind == HistoryItemKind.USER_PROMPT.value:
        return UserPrompt(
            id=history_item_db.id,
            history_id=history_item_db.history_id,
            created_at=history_item_db.created_at,
            prompt=history_item_db.content["prompt"],
        )
    elif history_item_db.kind == HistoryItemKind.MODEL_RESPONSE.value:
        return ModelResponse(
            id=history_item_db.id,
            history_id=history_item_db.history_id,
            created_at=history_item_db.created_at,
            response=history_item_db.content["response"],
        )
    elif history_item_db.kind == HistoryItemKind.THINKING_STEP.value:
        return ThinkingStep(
            id=history_item_db.id,
            history_id=history_item_db.history_id,
            created_at=history_item_db.created_at,
            thoughts=history_item_db.content["thoughts"],
        )
    elif history_item_db.kind == HistoryItemKind.TOOL_CALL.value:
        return ToolCall(
            id=history_item_db.id,
            history_id=history_item_db.history_id,
            created_at=history_item_db.created_at,
            tool_call_id=history_item_db.content["tool_call_id"],
            tool_name=history_item_db.content["tool_name"],
            args=history_item_db.content["args"],
        )
    elif history_item_db.kind == HistoryItemKind.TOOL_RESULT.value:
        return ToolResult(
            id=history_item_db.id,
            history_id=history_item_db.history_id,
            created_at=history_item_db.created_at,
            tool_call_id=history_item_db.content["tool_call_id"],
            tool_name=history_item_db.content["tool_name"],
            is_retry=history_item_db.content["is_retry"],
            result=history_item_db.content["result"],
        )
    else:
        raise ValueError(f"Unexpected history item: {history_item_db}")
