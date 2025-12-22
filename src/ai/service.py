from textwrap import dedent
from time import time_ns
from typing import AsyncIterator
from uuid import UUID, uuid4

import pydantic_ai.messages as paim
from pydantic_ai import Agent, AgentRun, FunctionToolset
from pydantic_ai.agent import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_ai.models.openai import OpenAIResponsesModel

# from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_graph.nodes import End as EndNode

from src.ai.mapper import PydanticAiMapper
from src.ai.models import (
    ModelResponseDelta,
    PartStart,
    StreamEnd,
    StreamItem,
    SystemPrompt,
    ThinkingDelta,
)
from src.history.service.models import HistoryItem, ModelResponse, ThinkingStep, UserPrompt
from src.history.service.service import HistoryService
from src.rag.service import RAGService


def dummy_tool(string: str) -> str:
    """Repeats the input text."""
    return string


dummy_tools = FunctionToolset(tools=[dummy_tool])


class AIService:
    # Streaming logic following https://ai.pydantic.dev/agents/
    def __init__(self, history_service: HistoryService, llm: OpenAIResponsesModel, rag_service: RAGService):
        self._llm = llm
        self._history_service = history_service
        self._rag_service = rag_service
        self._reset_state()

    def _reset_state(self):
        self._talked = ""
        self._thought = ""
        self._toolcalling = False
        self._thinking = False
        self._talking = False

    async def _handle_user_prompt_node(self, node: UserPromptNode, history_id: UUID) -> AsyncIterator[StreamItem]:  # type: ignore
        user_prompt = PydanticAiMapper.map_user_prompt_out(
            pai_user_prompt=node.user_prompt,
            id=uuid4(),
            history_id=history_id,
        )
        if user_prompt:
            await self._history_service.add_history_item(user_prompt)
            await self._rag_service.add_history_items([user_prompt])
            yield user_prompt

    async def _add_and_reset_talked(self, history_id: UUID) -> AsyncIterator[StreamItem]:
        self._talking = False
        if self._talked:
            model_response = ModelResponse(
                id=uuid4(),
                history_id=history_id,
                created_at=time_ns(),
                response=self._talked,
            )
            await self._history_service.add_history_item(model_response)
            await self._rag_service.add_history_items([model_response])
            self._talked = ""
            yield model_response

    async def _add_and_reset_thought(self, history_id: UUID) -> AsyncIterator[StreamItem]:
        self._thinking = False
        if self._thought:
            thinking_step = ThinkingStep(
                id=uuid4(),
                history_id=history_id,
                created_at=time_ns(),
                thoughts=self._thought,
            )
            await self._history_service.add_history_item(thinking_step)
            self._thought = ""
            yield thinking_step

    def _reset_toolcalling(self):
        self._toolcalling = False

    async def _handle_model_request_node(
        self,
        node: ModelRequestNode,  # type: ignore
        run: AgentRun,
        history_id: UUID,
    ) -> AsyncIterator[StreamItem]:
        # A model request node => We can stream tokens from the model's request
        async with node.stream(run.ctx) as request_stream:  # type: ignore
            final_result_found = False

            async for event in request_stream:
                if isinstance(event, paim.PartStartEvent):
                    pass
                elif isinstance(event, paim.PartDeltaEvent):
                    if isinstance(event.delta, paim.TextPartDelta):
                        async for item in self._add_and_reset_thought(history_id):
                            yield item
                        self._reset_toolcalling()
                        if not self._talking:
                            yield PartStart(
                                id=uuid4(),
                                created_at=time_ns(),
                                part_type="response",
                            )
                            self._talking = True
                        if event.delta.content_delta:
                            self._talked += event.delta.content_delta
                            yield ModelResponseDelta(
                                id=uuid4(),
                                created_at=time_ns(),
                                delta=event.delta.content_delta,
                            )

                    elif isinstance(event.delta, paim.ThinkingPartDelta):
                        async for item in self._add_and_reset_talked(history_id):
                            yield item
                        self._reset_toolcalling()
                        if not self._thinking:
                            yield PartStart(
                                id=uuid4(),
                                created_at=time_ns(),
                                part_type="thinking",
                            )
                            self._thinking = True
                        if event.delta.content_delta:
                            self._thought += event.delta.content_delta
                            yield ThinkingDelta(
                                id=uuid4(),
                                created_at=time_ns(),
                                delta=event.delta.content_delta,
                            )

                    elif isinstance(event.delta, paim.ToolCallPartDelta):  # type: ignore[reportUnnecessaryComparison]
                        async for item in self._add_and_reset_thought(history_id):
                            yield item
                        async for item in self._add_and_reset_talked(history_id):
                            yield item
                        if not self._toolcalling:
                            yield PartStart(
                                id=uuid4(),
                                created_at=time_ns(),
                                part_type="tool_call_prep",
                            )
                            self._toolcalling = True

                elif isinstance(event, paim.FinalResultEvent):
                    final_result_found = True
                    break

            if final_result_found:
                async for item in self._add_and_reset_thought(history_id):
                    yield item
                self._reset_toolcalling()
                async for item in self._add_and_reset_talked(history_id):
                    yield item
                yield PartStart(
                    id=uuid4(),
                    created_at=time_ns(),
                    part_type="final_response",
                )
                async for output in request_stream.stream_text(delta=True):
                    if output:
                        self._talked += output
                        yield ModelResponseDelta(
                            id=uuid4(),
                            created_at=time_ns(),
                            delta=output,
                        )
                async for item in self._add_and_reset_talked(history_id):
                    yield item

    async def _handle_call_tools_node(
        self,
        node: CallToolsNode,  # type: ignore
        run: AgentRun,
        history_id: UUID,
    ) -> AsyncIterator[StreamItem]:
        # A handle-response node => The model returned some data, potentially calls a tool
        async with node.stream(run.ctx) as handle_stream:  # type: ignore
            async for event in handle_stream:
                if isinstance(event, paim.FunctionToolCallEvent):
                    tool_call = PydanticAiMapper.map_tool_call_out(
                        pai_tool_call=event.part, id=uuid4(), history_id=history_id
                    )
                    await self._history_service.add_history_item(tool_call)
                    yield tool_call
                elif isinstance(event, paim.FunctionToolResultEvent):
                    tool_result = PydanticAiMapper.map_tool_result_out(
                        pai_tool_result=event.result, id=uuid4(), history_id=history_id
                    )
                    await self._history_service.add_history_item(tool_result)
                    yield tool_result

    async def _handle_end_node(self, node: EndNode, run: AgentRun, history_id: UUID) -> AsyncIterator[StreamItem]:  # type: ignore
        yield StreamEnd(id=uuid4(), created_at=time_ns())

    async def stream_agent_run(
        self,
        user_prompt: UserPrompt,
        last_n_history_items: int = 10,
        n_memory_items: int = 10,
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

        main_system_prompt = self._main_system_prompt()
        history_items.insert(0, main_system_prompt)
        memory_prompt = await self._rag_service.search_for_user_prompt(
            user_prompt=user_prompt,
            top_k=n_memory_items,
        )
        history_items.insert(0, memory_prompt)

        pai_history = PydanticAiMapper.map_history_items_in(history_items)

        agent = Agent(model=self._llm, toolsets=[dummy_tools])
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
        self._reset_state()

    def _main_system_prompt(self) -> SystemPrompt:
        return SystemPrompt(
            id=uuid4(),
            created_at=time_ns(),
            prompt=dedent("""
                [# General Instructions #]

                You are a helpful assistant .
            """),
        )
