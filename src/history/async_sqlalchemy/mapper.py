from src.history.async_sqlalchemy.models import HistoryItemDb
from src.history.models import (
    HistoryItem,
    HistoryItemKind,
    ModelResponse,
    ThinkingStep,
    ToolCall,
    ToolResult,
    UserPrompt,
)


def map_history_item_to_db(history_item: HistoryItem) -> HistoryItemDb:
    match history_item:
        case UserPrompt():
            return HistoryItemDb(
                id=history_item.id,
                history_id=history_item.history_id,
                created_at=history_item.created_at,
                kind=HistoryItemKind.USER_PROMPT.value,
                content={"prompt": history_item.prompt},
            )
        case ModelResponse():
            return HistoryItemDb(
                id=history_item.id,
                history_id=history_item.history_id,
                created_at=history_item.created_at,
                kind=HistoryItemKind.MODEL_RESPONSE.value,
                content={"response": history_item.response},
            )
        case ThinkingStep():
            return HistoryItemDb(
                id=history_item.id,
                history_id=history_item.history_id,
                created_at=history_item.created_at,
                kind=HistoryItemKind.THINKING_STEP.value,
                content={"thoughts": history_item.thoughts},
            )
        case ToolCall():
            return HistoryItemDb(
                id=history_item.id,
                history_id=history_item.history_id,
                created_at=history_item.created_at,
                kind=HistoryItemKind.TOOL_CALL.value,
                content={
                    "tool_call_id": history_item.tool_call_id,
                    "tool_name": history_item.tool_name,
                    "args": history_item.args,
                },
            )
        case ToolResult():
            return HistoryItemDb(
                id=history_item.id,
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


def map_history_item_to_domain(history_item_db: HistoryItemDb) -> HistoryItem:
    match history_item_db.kind:
        case HistoryItemKind.USER_PROMPT.value:
            return UserPrompt(
                id=history_item_db.id,
                history_id=history_item_db.history_id,
                created_at=history_item_db.created_at,
                prompt=history_item_db.content["prompt"],
            )
        case HistoryItemKind.MODEL_RESPONSE.value:
            return ModelResponse(
                id=history_item_db.id,
                history_id=history_item_db.history_id,
                created_at=history_item_db.created_at,
                response=history_item_db.content["response"],
            )
        case HistoryItemKind.THINKING_STEP.value:
            return ThinkingStep(
                id=history_item_db.id,
                history_id=history_item_db.history_id,
                created_at=history_item_db.created_at,
                thoughts=history_item_db.content["thoughts"],
            )
        case HistoryItemKind.TOOL_CALL.value:
            return ToolCall(
                id=history_item_db.id,
                history_id=history_item_db.history_id,
                created_at=history_item_db.created_at,
                tool_call_id=history_item_db.content["tool_call_id"],
                tool_name=history_item_db.content["tool_name"],
                args=history_item_db.content["args"],
            )
        case HistoryItemKind.TOOL_RESULT.value:
            return ToolResult(
                id=history_item_db.id,
                history_id=history_item_db.history_id,
                created_at=history_item_db.created_at,
                tool_call_id=history_item_db.content["tool_call_id"],
                tool_name=history_item_db.content["tool_name"],
                is_retry=history_item_db.content["is_retry"],
                result=history_item_db.content["result"],
            )
        case _:
            raise ValueError(f"Unexpected history item: {history_item_db}")
