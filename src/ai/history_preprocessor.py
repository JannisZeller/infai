from src.ai.models import SystemPrompt
from src.history.models import HistoryItem, ModelResponse, ThinkingStep, ToolCall, ToolResult, UserPrompt


def preprocess_history(history: list[HistoryItem | SystemPrompt]) -> list[HistoryItem | SystemPrompt]:
    """Preprocesses the history to remove orphan ToolCalls and ToolResults. Also, ToolResults
    are sorted after the ToolCalls to avoid Pydantic AI from failing to process the history.

    Args:
        history: list[HistoryItem | SystemPrompt] - The history to preprocess.

    Returns:
        list[HistoryItem | SystemPrompt] - The preprocessed history.
    """

    preprocessed_history: list[HistoryItem | SystemPrompt] = []
    tool_call_ids_to_results: dict[str, ToolResult] = {}

    for tool_result in history:
        # Extracting ToolResults up front to avoid having to iterate over the history
        # for each ToolCall in the second pass through.
        match tool_result:
            case ToolResult():
                tool_call_ids_to_results[tool_result.tool_call_id] = tool_result
            case _:
                pass

    for item in history:
        match item:
            case SystemPrompt():
                preprocessed_history.append(item)
            case UserPrompt():
                preprocessed_history.append(item)
            case ThinkingStep():
                # Skipping thinking steps
                pass
            case ModelResponse():
                preprocessed_history.append(item)
            case ToolCall():
                tool_call_id = item.tool_call_id
                if tool_call_id in tool_call_ids_to_results:
                    result = tool_call_ids_to_results[tool_call_id]
                    # Only add the ToolCall if there is a result for it
                    preprocessed_history.append(item)
                    # Add the ToolResult directly after the ToolCall
                    # otherwise, Pydantic AI cannot process the
                    # history after mapping it in.
                    preprocessed_history.append(result)
            case ToolResult():
                # Gets added with the ToolCall
                pass

    return preprocessed_history
