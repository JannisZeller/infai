import logging
from typing import Sequence

import pydantic_ai.messages as paim

from src.ai.models import ModelResponseDelta, StreamItem, SystemPrompt, ThinkingDelta
from src.history.models import History, HistoryItem, ModelResponse, ThinkingStep, ToolCall, ToolResult, UserPrompt

logger = logging.getLogger(__name__)


class PydanticAiMapper:
    # System Prompt
    # --------------------------------------------------------------------------------
    @staticmethod
    def _map_system_prompt_out(pai_system_prompt: paim.SystemPromptPart) -> SystemPrompt:
        return SystemPrompt(content=pai_system_prompt.content)

    @staticmethod
    def _map_system_prompt_in(system_prompt: SystemPrompt) -> paim.ModelRequest:
        return paim.ModelRequest(parts=[paim.SystemPromptPart(content=system_prompt.content)])

    # User Prompt
    # --------------------------------------------------------------------------------
    @staticmethod
    def map_user_prompt_out(pai_user_prompt: str | Sequence[paim.UserContent] | None) -> UserPrompt | None:
        if isinstance(pai_user_prompt, str):
            return UserPrompt(prompt=pai_user_prompt)
        if isinstance(pai_user_prompt, Sequence):
            full_user_prompt = ""
            for user_prompt_part in pai_user_prompt:
                if isinstance(user_prompt_part, str):
                    full_user_prompt += user_prompt_part
                else:
                    logger.warning(f"Unexpected user prompt part: {user_prompt_part} - skipping.")
                    continue
            return UserPrompt(prompt=full_user_prompt)
        else:
            logger.warning(f"Unexpected PydanticAI UserPromptNode: {pai_user_prompt} - skipping.")
            return None

    @staticmethod
    def _map_user_prompt_in(user_prompt: UserPrompt) -> paim.ModelRequest:
        return paim.ModelRequest(parts=[paim.UserPromptPart(content=user_prompt.prompt)])

    # Thinking Part
    # --------------------------------------------------------------------------------
    @staticmethod
    def _map_thinking_delta_out(pai_thinking_delta: paim.ThinkingPartDelta) -> ThinkingDelta:
        if not pai_thinking_delta.content_delta:
            logger.warning(f"Unexpected thinking delta part: {pai_thinking_delta} using empty.")
            return ThinkingDelta(delta="")
        return ThinkingDelta(delta=pai_thinking_delta.content_delta)

    @staticmethod
    def _map_thinking_step_out(pai_thinking_step: paim.ThinkingPart) -> ThinkingStep:
        return ThinkingStep(thoughts=pai_thinking_step.content)

    @staticmethod
    def _map_thinking_step_in(thinking_step: ThinkingStep) -> paim.ModelResponse:
        return paim.ModelResponse(parts=[paim.ThinkingPart(content=thinking_step.thoughts)])

    # Tool Call
    # --------------------------------------------------------------------------------
    @staticmethod
    def map_tool_call_out(pai_tool_call: paim.ToolCallPart) -> ToolCall:
        return ToolCall(
            tool_name=pai_tool_call.tool_name,
            tool_call_id=pai_tool_call.tool_call_id,
            args=pai_tool_call.args,
        )

    @staticmethod
    def _map_tool_call_in(tool_call: ToolCall) -> paim.ModelResponse:
        return paim.ModelResponse(
            parts=[
                paim.ToolCallPart(
                    tool_name=tool_call.tool_name,
                    tool_call_id=tool_call.tool_call_id,
                    args=tool_call.args or {},
                )
            ]
        )

    # Tool Result
    # --------------------------------------------------------------------------------
    @staticmethod
    def _map_tool_result_out(pai_tool_result: paim.ToolReturnPart) -> ToolResult:
        return ToolResult(
            tool_name=pai_tool_result.tool_name,
            tool_call_id=pai_tool_result.tool_call_id,
            is_retry=False,
            result=pai_tool_result.content,
        )

    @staticmethod
    def _map_retry_part_out(pai_retry_part: paim.RetryPromptPart) -> ToolResult:
        return ToolResult(
            tool_name=pai_retry_part.tool_name or "unknown",
            tool_call_id=pai_retry_part.tool_call_id,
            is_retry=True,
            result=pai_retry_part.content,
        )

    @staticmethod
    def map_tool_result_out(pai_tool_result: paim.ToolReturnPart | paim.RetryPromptPart) -> ToolResult:
        if isinstance(pai_tool_result, paim.ToolReturnPart):
            return PydanticAiMapper._map_tool_result_out(pai_tool_result)
        elif isinstance(pai_tool_result, paim.RetryPromptPart):  # type: ignore - to be explicit
            return PydanticAiMapper._map_retry_part_out(pai_tool_result)

    @staticmethod
    def _map_tool_result_in(tool_result: ToolResult) -> paim.ModelRequest:
        if tool_result.is_retry:
            return paim.ModelRequest(
                parts=[
                    paim.RetryPromptPart(
                        tool_name=tool_result.tool_name,
                        tool_call_id=tool_result.tool_call_id,
                        content=tool_result.result,
                    )
                ]
            )
        return paim.ModelRequest(
            parts=[
                paim.ToolReturnPart(
                    tool_name=tool_result.tool_name,
                    tool_call_id=tool_result.tool_call_id,
                    content=tool_result.result,
                )
            ]
        )

    # Model Response
    # --------------------------------------------------------------------------------
    @staticmethod
    def _map_model_response_delta_out(pai_model_response_delta: paim.TextPartDelta) -> ModelResponseDelta:
        return ModelResponseDelta(delta=pai_model_response_delta.content_delta)

    @staticmethod
    def _map_model_response_out(pai_model_response: paim.TextPart) -> ModelResponse:
        return ModelResponse(response=pai_model_response.content)

    @staticmethod
    def _map_model_response_in(model_response: ModelResponse) -> paim.ModelResponse:
        return paim.ModelResponse(parts=[paim.TextPart(content=model_response.response)])

    # Any
    # --------------------------------------------------------------------------------

    @staticmethod
    def _map_model_message_out(paim_item: paim.ModelMessage) -> list[StreamItem]:
        stream_items: list[StreamItem] = []
        if isinstance(paim_item, paim.ModelRequest):
            for part in paim_item.parts:
                if isinstance(part, paim.SystemPromptPart):
                    stream_items.append(PydanticAiMapper._map_system_prompt_out(part))
                elif isinstance(part, paim.UserPromptPart):
                    user_prompt = PydanticAiMapper.map_user_prompt_out(part.content)
                    if user_prompt:
                        stream_items.append(user_prompt)
                elif isinstance(part, paim.ToolReturnPart):
                    stream_items.append(PydanticAiMapper._map_tool_result_out(part))
                elif isinstance(part, paim.RetryPromptPart):  # type: ignore[reportUnnecessaryComparison]
                    stream_items.append(PydanticAiMapper._map_retry_part_out(part))
        else:
            for part in paim_item.parts:
                if isinstance(part, paim.TextPartDelta):
                    stream_items.append(PydanticAiMapper._map_model_response_delta_out(part))
                elif isinstance(part, paim.TextPart):
                    stream_items.append(PydanticAiMapper._map_model_response_out(part))
                elif isinstance(part, paim.ThinkingPart):
                    stream_items.append(PydanticAiMapper._map_thinking_step_out(part))
                elif isinstance(part, paim.ThinkingPartDelta):
                    stream_items.append(PydanticAiMapper._map_thinking_delta_out(part))
                elif isinstance(part, paim.ToolCallPart):
                    stream_items.append(PydanticAiMapper.map_tool_call_out(part))
                else:
                    logger.warning(f"Unexpected history item: {part} skipping.")
        return stream_items

    @staticmethod
    def _map_history_item_in(history_item: HistoryItem) -> paim.ModelMessage | None:
        if isinstance(history_item, SystemPrompt):
            return PydanticAiMapper._map_system_prompt_in(history_item)
        elif isinstance(history_item, UserPrompt):
            return PydanticAiMapper._map_user_prompt_in(history_item)
        elif isinstance(history_item, ThinkingStep):
            return PydanticAiMapper._map_thinking_step_in(history_item)
        elif isinstance(history_item, ToolCall):
            return PydanticAiMapper._map_tool_call_in(history_item)
        elif isinstance(history_item, ToolResult):
            return PydanticAiMapper._map_tool_result_in(history_item)
        elif isinstance(history_item, ModelResponse):  # type: ignore - staying explicit
            return PydanticAiMapper._map_model_response_in(history_item)

    # History
    # --------------------------------------------------------------------------------
    @staticmethod
    def map_history_in(history: History) -> list[paim.ModelRequest | paim.ModelResponse]:
        pai_history: list[paim.ModelRequest | paim.ModelResponse] = []
        for item in history.items:
            paim_message = PydanticAiMapper._map_history_item_in(item)
            if paim_message:
                pai_history.append(paim_message)
            else:
                logger.warning(f"Unexpected history item: {item} skipping.")
        return pai_history
