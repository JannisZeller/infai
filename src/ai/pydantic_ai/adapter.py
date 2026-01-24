from time import time_ns
from typing import Any, AsyncIterator, Callable
from uuid import UUID, uuid4

import pydantic_ai.messages as paim
from pydantic_ai import Agent, AgentRun
from pydantic_ai.agent import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_graph.nodes import End as EndNode

from src.ai.history_preprocessor import preprocess_history
from src.ai.model_request_yields import ModelRequestCurrentPart, PartState
from src.ai.models import (
    StreamEnd,
    StreamItem,
    SystemPrompt,
)
from src.ai.prompts import PromptsService
from src.ai.pydantic_ai.mapper import PydanticAiMapper
from src.ai.pydantic_ai.tools import PydanticAIToolProvider
from src.config.models import Config
from src.core.logging import get_logger
from src.history.models import HistoryItem, UserPrompt
from src.history.service import HistoryService
from src.rag.port import RAGService
from src.tools.models import ToolSet

logger = get_logger(__name__, output="console")


def dummy_tool(string: str) -> str:
    """Repeats the input text."""
    return string


class PydanticAIService:
    # Streaming logic following https://ai.pydantic.dev/agents/
    def __init__(
        self,
        config: Config,
        llm: OpenAIResponsesModel | OpenAIChatModel,
        history_service: HistoryService,
        rag_service: RAGService | None,
        prompts_service: PromptsService,
    ):
        self._llm = llm
        self._history_service = history_service
        self._rag_service = rag_service
        self._prompts_service = prompts_service

    async def _handle_user_prompt_node(self, node: UserPromptNode, history_id: UUID) -> AsyncIterator[StreamItem]:  # type: ignore
        user_prompt = PydanticAiMapper.map_user_prompt_out(
            pai_user_prompt=node.user_prompt,
            id=uuid4(),
            history_id=history_id,
        )
        if user_prompt:
            await self._history_service.add_history_item(user_prompt)
            if self._rag_service:
                await self._rag_service.add_history_items([user_prompt])
            yield user_prompt

    def _try_extracting_raw_content_start_of_part(
        self,
        current_part: ModelRequestCurrentPart,
        event: paim.PartStartEvent,
    ):
        # Best effort to handle provider_details for not standard OpenAI APIs.
        try:
            if current_part.provider_details and not event.part.has_content():
                raw_content: list[str] = current_part.provider_details["raw_content"]
                if raw_content:
                    return current_part.add_content_and_yield_delta(content="".join(raw_content))
        except Exception as exc:
            logger.warning(f"Failed to extract raw content from provider_details: {exc}")

    def _try_extracting_raw_content_delta(
        self,
        current_part: ModelRequestCurrentPart,
        provider_details: Callable[[dict[str, Any]], dict[str, Any]],
    ):
        # Best effort to handle provider_details-callable for not standard OpenAI APIs.
        try:
            assert current_part.provider_details, "provider_details must be set"
            new_provider_details = provider_details(current_part.provider_details)
            delta = new_provider_details["raw_content"][-1]
            if delta:
                return current_part.add_content_and_yield_delta(content=delta)
        except Exception as exc:
            logger.warning(f"Failed to extract raw content from provider_details (callable): {exc}")
            return None

    async def _handle_model_request_node(
        self,
        node: ModelRequestNode,  # type: ignore
        run: AgentRun,
        history_id: UUID,
    ) -> AsyncIterator[StreamItem]:
        # A model request node => We can stream tokens from the model's request
        async with node.stream(run.ctx) as request_stream:  # type: ignore
            current_part = ModelRequestCurrentPart(history_id=history_id)

            async for event in request_stream:
                match event:
                    case paim.PartStartEvent():
                        match event.part:
                            case paim.ThinkingPart():
                                separator = "\n\n" if current_part.state == PartState.THINKING else ""
                                # If we are switching from a different type, flush the previous part
                                # and start tracking thinking part
                                if current_part.is_streaming_but_not_in_state(PartState.THINKING):
                                    if flushed_part := current_part.flush():
                                        yield flushed_part
                                    yield current_part.reset_to_state_and_get_part_start(PartState.THINKING)
                                # If we are not currently streaming, reset the part to thinking
                                # and yield the part start event.
                                elif current_part.is_not_streaming():
                                    yield current_part.reset_to_state_and_get_part_start(PartState.THINKING)
                                # Add content and yield delta if present using a separator to fix the formatting.
                                # Multiple thinking parts might be in a row.
                                # Other than for the TextPart below we always yield
                                # because we at least have the separator if there's content.
                                # Only yield if there's actual content to avoid empty deltas.
                                content = separator + event.part.content
                                if content:
                                    yield current_part.add_content_and_yield_delta(content=content)
                                else:
                                    current_part.provider_details = event.part.provider_details
                                    if current_part.provider_details:
                                        extracted_content = self._try_extracting_raw_content_start_of_part(
                                            current_part=current_part,
                                            event=event,
                                        )
                                        if extracted_content:
                                            yield extracted_content

                            case paim.TextPart():
                                if current_part.is_streaming_but_not_in_state(PartState.TALKING):
                                    if flushed_part := current_part.flush():
                                        yield flushed_part
                                    yield current_part.reset_to_state_and_get_part_start(PartState.TALKING)

                                elif current_part.is_not_streaming():
                                    yield current_part.reset_to_state_and_get_part_start(PartState.TALKING)

                                if event.part.has_content():
                                    yield current_part.add_content_and_yield_delta(event.part.content)

                                else:
                                    current_part.provider_details = event.part.provider_details
                                    if current_part.provider_details:
                                        extracted_content = self._try_extracting_raw_content_start_of_part(
                                            current_part=current_part,
                                            event=event,
                                        )
                                        if extracted_content:
                                            yield extracted_content

                            case paim.ToolCallPart():
                                # Special handling: TOOL_CALL_PREP is a state for (potentially) multiple tool calls
                                if current_part.state == PartState.TOOL_CALL_PREP:
                                    # Already in tool call prep mode - stay there for parallel calls
                                    pass
                                else:
                                    # Transitioning to tool call prep - flush any previous content, same as above.
                                    if current_part.is_streaming_but_not_in_state(PartState.NO_STREAM):
                                        if flushed_part := current_part.flush():
                                            yield flushed_part
                                    yield current_part.reset_to_state_and_get_part_start(PartState.TOOL_CALL_PREP)

                            case paim.BuiltinToolCallPart() | paim.BuiltinToolReturnPart() | paim.FilePart():
                                if current_part.is_streaming_but_not_in_state(PartState.NO_STREAM):
                                    if flushed_part := current_part.flush():
                                        yield flushed_part
                                    current_part.reset_to_no_stream()

                    case paim.PartDeltaEvent():
                        match event.delta:
                            case paim.ThinkingPartDelta():
                                if event.delta.content_delta:
                                    yield current_part.add_content_and_yield_delta(content=event.delta.content_delta)

                                elif event.delta.provider_details:
                                    if callable(event.delta.provider_details):
                                        extracted_content = self._try_extracting_raw_content_delta(
                                            current_part=current_part,
                                            provider_details=event.delta.provider_details,  # type: ignore
                                        )
                                        if extracted_content:
                                            yield extracted_content
                                    else:
                                        logger.error("Processing provider_details of type dict is not supported yet.")

                            case paim.TextPartDelta():
                                if event.delta.content_delta:
                                    yield current_part.add_content_and_yield_delta(content=event.delta.content_delta)

                                elif event.delta.provider_details:
                                    if callable(event.delta.provider_details):
                                        extracted_content = self._try_extracting_raw_content_delta(
                                            current_part=current_part,
                                            provider_details=event.delta.provider_details,  # type: ignore
                                        )
                                        if extracted_content:
                                            yield extracted_content
                                    else:
                                        logger.error("Processing provider_details of type dict is not supported yet.")

                            case paim.ToolCallPartDelta():
                                pass

                    case paim.PartEndEvent():
                        # Do not flush on PartEndEvent - we want to collapse consecutive parts of the same type.
                        pass

                    case paim.FinalResultEvent():
                        # Currently, streaming structured output is not supported, we use the TextPartDeltas directly.
                        pass

    async def _handle_call_tools_node(
        self,
        node: CallToolsNode,  # type: ignore
        run: AgentRun,
        history_id: UUID,
    ) -> AsyncIterator[StreamItem]:
        # A handle-response node => The model returned some data, potentially calls a tool
        async with node.stream(run.ctx) as handle_stream:  # type: ignore
            async for event in handle_stream:
                match event:
                    case paim.FunctionToolCallEvent():
                        tool_call = PydanticAiMapper.map_tool_call_out(
                            pai_tool_call=event.part,
                            id=uuid4(),
                            history_id=history_id,
                        )
                        await self._history_service.add_history_item(tool_call)
                        yield tool_call
                    case paim.FunctionToolResultEvent():
                        tool_result = PydanticAiMapper.map_tool_result_out(
                            pai_tool_result=event.result,
                            id=uuid4(),
                            history_id=history_id,
                        )
                        await self._history_service.add_history_item(tool_result)
                        yield tool_result
                    case _:
                        logger.warning(f"Unexpected CallToolsNode event in AI execution loop: {event} skipping.")

    async def _handle_end_node(self, node: EndNode, run: AgentRun, history_id: UUID) -> AsyncIterator[StreamItem]:  # type: ignore
        yield StreamEnd(id=uuid4(), history_id=history_id, created_at=time_ns())

    async def stream_agent_run(
        self,
        user_prompt: UserPrompt,
        last_n_history_items: int = 10,
        n_memory_items: int = 10,
        tool_sets: list[ToolSet] = [],
    ) -> AsyncIterator[StreamItem]:
        """
        Stream the agent run, yielding StreamItems that can be consumed by UI services.
        """
        pai_user_prompt = user_prompt.prompt
        history_id = user_prompt.history_id

        history_items: list[HistoryItem | SystemPrompt] = list(
            await self._history_service.get_last_n_history_items(
                history_id=history_id,
                n=last_n_history_items,
            )
        )

        main_system_prompt = self._prompts_service.get_system_prompt(
            history_id=history_id,
            tool_sets=tool_sets,
        )
        history_items.insert(0, main_system_prompt)

        if self._rag_service:
            memory_prompt = await self._rag_service.search_for_user_prompt(
                user_prompt=user_prompt,
                top_k=n_memory_items,
            )
            history_items.insert(0, memory_prompt)

        history_items = preprocess_history(history_items)

        pai_history = PydanticAiMapper.map_history_items_in(history_items)

        pai_toolsets = [PydanticAIToolProvider.get_pai_toolset(tool_set) for tool_set in tool_sets]

        agent = Agent(model=self._llm, toolsets=pai_toolsets)
        async with agent.iter(pai_user_prompt, message_history=pai_history) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node):
                    async for item in self._handle_user_prompt_node(node=node, history_id=history_id):  # type: ignore
                        yield item
                elif Agent.is_model_request_node(node):
                    async for item in self._handle_model_request_node(node=node, run=run, history_id=history_id):  # type: ignore
                        yield item
                elif Agent.is_call_tools_node(node):
                    async for item in self._handle_call_tools_node(node=node, run=run, history_id=history_id):  # type: ignore
                        yield item
                elif Agent.is_end_node(node):
                    async for item in self._handle_end_node(node=node, run=run, history_id=history_id):  # type: ignore
                        yield item
