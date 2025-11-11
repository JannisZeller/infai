from time import time_ns
from typing import Sequence
from uuid import UUID

import pydantic_ai.messages as paim

from src.ai.models import ModelResponseDelta, StreamItem, SystemPrompt, ThinkingDelta
from src.core.logging import get_logger
from src.history.models import (
    HistoryItem,
    ModelResponse,
    ThinkingStep,
    ToolCall,
    ToolResult,
    UserPrompt,
)

logger = get_logger(__name__, output="file")


class PydanticAiMapper:
    # System Prompt
    # --------------------------------------------------------------------------------
    @staticmethod
    def _map_system_prompt_out(pai_system_prompt: paim.SystemPromptPart, id: UUID, history_id: UUID) -> SystemPrompt:
        return SystemPrompt(
            id=id,
            history_id=history_id,
            created_at=time_ns(),
            prompt=pai_system_prompt.content,
        )

    @staticmethod
    def _map_system_prompt_in(system_prompt: SystemPrompt) -> paim.ModelRequest:
        return paim.ModelRequest(parts=[paim.SystemPromptPart(content=system_prompt.prompt)])

    # User Prompt
    # --------------------------------------------------------------------------------
    @staticmethod
    def map_user_prompt_out(
        pai_user_prompt: str | Sequence[paim.UserContent] | None,
        id: UUID,
        history_id: UUID,
    ) -> UserPrompt | None:
        match pai_user_prompt:
            case str():
                return UserPrompt(
                    id=id,
                    created_at=time_ns(),
                    history_id=history_id,
                    prompt=pai_user_prompt,
                )
            case Sequence():
                full_user_prompt = ""
                for user_prompt_part in pai_user_prompt:
                    if isinstance(user_prompt_part, str):
                        full_user_prompt += user_prompt_part
                    else:
                        logger.warning(f"Unexpected user prompt part: {user_prompt_part} - skipping.")
                        continue
                return UserPrompt(
                    id=id,
                    created_at=time_ns(),
                    history_id=history_id,
                    prompt=full_user_prompt,
                )
            case _:
                logger.warning(f"Unexpected PydanticAI UserPromptNode: {pai_user_prompt} - skipping.")
                return None

    @staticmethod
    def _map_user_prompt_in(user_prompt: UserPrompt) -> paim.ModelRequest:
        return paim.ModelRequest(parts=[paim.UserPromptPart(content=user_prompt.prompt)])

    # Thinking Part
    # --------------------------------------------------------------------------------
    @staticmethod
    def _map_thinking_delta_out(
        pai_thinking_delta: paim.ThinkingPartDelta, id: UUID, history_id: UUID
    ) -> ThinkingDelta:
        if not pai_thinking_delta.content_delta:
            logger.warning(f"Unexpected thinking delta part: {pai_thinking_delta} using empty.")
            return ThinkingDelta(
                id=id,
                history_id=history_id,
                created_at=time_ns(),
                delta="",
            )
        return ThinkingDelta(
            id=id,
            history_id=history_id,
            created_at=time_ns(),
            delta=pai_thinking_delta.content_delta,
        )

    @staticmethod
    def _map_thinking_step_out(
        pai_thinking_step: paim.ThinkingPart,
        id: UUID,
        history_id: UUID,
    ) -> ThinkingStep:
        return ThinkingStep(
            id=id,
            history_id=history_id,
            created_at=time_ns(),
            thoughts=pai_thinking_step.content,
        )

    @staticmethod
    def _map_thinking_step_in(thinking_step: ThinkingStep) -> paim.ModelResponse:
        return paim.ModelResponse(parts=[paim.ThinkingPart(content=thinking_step.thoughts)])

    # Tool Call
    # --------------------------------------------------------------------------------
    @staticmethod
    def map_tool_call_out(
        pai_tool_call: paim.ToolCallPart,
        id: UUID,
        history_id: UUID,
    ) -> ToolCall:
        return ToolCall(
            id=id,
            created_at=time_ns(),
            history_id=history_id,
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
    def _map_tool_result_out(
        pai_tool_result: paim.ToolReturnPart,
        id: UUID,
        history_id: UUID,
    ) -> ToolResult:
        return ToolResult(
            id=id,
            created_at=time_ns(),
            history_id=history_id,
            tool_name=pai_tool_result.tool_name,
            tool_call_id=pai_tool_result.tool_call_id,
            is_retry=False,
            result=pai_tool_result.content,
        )

    @staticmethod
    def _map_retry_part_out(
        pai_retry_part: paim.RetryPromptPart,
        id: UUID,
        history_id: UUID,
    ) -> ToolResult:
        return ToolResult(
            id=id,
            created_at=time_ns(),
            history_id=history_id,
            tool_name=pai_retry_part.tool_name or "unknown",
            tool_call_id=pai_retry_part.tool_call_id,
            is_retry=True,
            result=pai_retry_part.content,
        )

    @staticmethod
    def map_tool_result_out(
        pai_tool_result: paim.ToolReturnPart | paim.RetryPromptPart,
        id: UUID,
        history_id: UUID,
    ) -> ToolResult:
        match pai_tool_result:
            case paim.ToolReturnPart():
                return PydanticAiMapper._map_tool_result_out(
                    pai_tool_result=pai_tool_result,
                    id=id,
                    history_id=history_id,
                )
            case paim.RetryPromptPart():
                return PydanticAiMapper._map_retry_part_out(
                    pai_retry_part=pai_tool_result,
                    id=id,
                    history_id=history_id,
                )

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
    def _map_model_response_delta_out(
        pai_model_response_delta: paim.TextPartDelta,
        id: UUID,
        history_id: UUID,
    ) -> ModelResponseDelta:
        return ModelResponseDelta(
            id=id,
            history_id=history_id,
            created_at=time_ns(),
            delta=pai_model_response_delta.content_delta,
        )

    @staticmethod
    def _map_model_response_out(
        pai_model_response: paim.TextPart,
        id: UUID,
        history_id: UUID,
    ) -> ModelResponse:
        return ModelResponse(
            id=id,
            history_id=history_id,
            created_at=time_ns(),
            response=pai_model_response.content,
        )

    @staticmethod
    def _map_model_response_in(model_response: ModelResponse) -> paim.ModelResponse:
        return paim.ModelResponse(parts=[paim.TextPart(content=model_response.response)])

    # Any
    # --------------------------------------------------------------------------------

    @staticmethod
    def _map_model_message_out(
        paim_item: paim.ModelMessage,
        id: UUID,
        history_id: UUID,
    ) -> list[StreamItem]:
        stream_items: list[StreamItem] = []
        match paim_item:
            case paim.ModelRequest():
                for part in paim_item.parts:
                    match part:
                        case paim.SystemPromptPart():
                            stream_items.append(
                                PydanticAiMapper._map_system_prompt_out(
                                    pai_system_prompt=part,
                                    id=id,
                                    history_id=history_id,
                                )
                            )
                        case paim.UserPromptPart():
                            user_prompt = PydanticAiMapper.map_user_prompt_out(
                                pai_user_prompt=part.content,
                                id=id,
                                history_id=history_id,
                            )
                            if user_prompt:
                                stream_items.append(user_prompt)
                        case paim.ToolReturnPart():
                            stream_items.append(
                                PydanticAiMapper._map_tool_result_out(
                                    pai_tool_result=part,
                                    id=id,
                                    history_id=history_id,
                                )
                            )
                        case paim.RetryPromptPart():
                            stream_items.append(
                                PydanticAiMapper._map_retry_part_out(
                                    pai_retry_part=part,
                                    id=id,
                                    history_id=history_id,
                                )
                            )
            case paim.ModelResponse():
                for part in paim_item.parts:
                    if isinstance(part, paim.TextPartDelta):
                        stream_items.append(
                            PydanticAiMapper._map_model_response_delta_out(
                                pai_model_response_delta=part,
                                id=id,
                                history_id=history_id,
                            )
                        )
                    elif isinstance(part, paim.TextPart):
                        stream_items.append(
                            PydanticAiMapper._map_model_response_out(
                                pai_model_response=part,
                                id=id,
                                history_id=history_id,
                            )
                        )
                    elif isinstance(part, paim.ThinkingPart):
                        stream_items.append(
                            PydanticAiMapper._map_thinking_step_out(
                                pai_thinking_step=part,
                                id=id,
                                history_id=history_id,
                            )
                        )
                    elif isinstance(part, paim.ThinkingPartDelta):
                        stream_items.append(
                            PydanticAiMapper._map_thinking_delta_out(
                                pai_thinking_delta=part,
                                id=id,
                                history_id=history_id,
                            )
                        )
                    elif isinstance(part, paim.ToolCallPart):
                        stream_items.append(
                            PydanticAiMapper.map_tool_call_out(
                                pai_tool_call=part,
                                id=id,
                                history_id=history_id,
                            )
                        )
                    else:
                        logger.warning(f"Unexpected history item: {part} skipping.")
        return stream_items

    @staticmethod
    def _map_history_item_in(history_item: HistoryItem | SystemPrompt) -> paim.ModelMessage | None:
        match history_item:
            case SystemPrompt():
                return PydanticAiMapper._map_system_prompt_in(history_item)
            case UserPrompt():
                return PydanticAiMapper._map_user_prompt_in(history_item)
            case ThinkingStep():
                return PydanticAiMapper._map_thinking_step_in(history_item)
            case ToolCall():
                return PydanticAiMapper._map_tool_call_in(history_item)
            case ToolResult():
                return PydanticAiMapper._map_tool_result_in(history_item)
            case ModelResponse():
                return PydanticAiMapper._map_model_response_in(history_item)

    # History
    # --------------------------------------------------------------------------------
    @staticmethod
    def map_history_items_in(
        history_items: list[HistoryItem | SystemPrompt],
    ) -> list[paim.ModelRequest | paim.ModelResponse]:
        pai_history: list[paim.ModelRequest | paim.ModelResponse] = []
        for item in history_items:
            paim_message = PydanticAiMapper._map_history_item_in(item)
            if paim_message:
                pai_history.append(paim_message)
            else:
                logger.warning(f"Unexpected history item: {item} skipping.")
        return pai_history
