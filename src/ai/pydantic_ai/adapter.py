from dataclasses import dataclass
from time import time_ns
from typing import Any, AsyncIterator, Sequence
from uuid import UUID, uuid4

import pydantic_ai.messages as paim
from pydantic_ai import Agent, AgentRun, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.agent import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_graph.nodes import End as EndNode

from src.ai.history_preprocessor import preprocess_history
from src.ai.model_request_yields import ModelRequestCurrentPart, PartState
from src.ai.models import (
    StreamEnd,
    StreamItem,
    SystemPrompt,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from src.ai.prompts import PromptsService
from src.ai.pydantic_ai.mapper import PydanticAiMapper
from src.ai.pydantic_ai.tools import PydanticAIToolProvider
from src.config.models import Config
from src.history.models import HistoryItem, UserPrompt
from src.history.service import HistoryService
from src.rag.port import RAGService
from src.tools.models import ToolSet

PENDING_APPROVAL_TTL_NS = 15 * 60 * 1_000_000_000
MAX_PENDING_APPROVAL_RUNS = 100


@dataclass(frozen=True)
class PendingApprovalRun:
    created_at: int
    history_id: UUID
    pai_history: list[paim.ModelMessage]
    tool_sets: list[ToolSet]
    approval_tool_call_ids: set[str]
    metadata: dict[str, dict[str, Any]]


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
        self._config = config
        self._llm = llm
        self._history_service = history_service
        self._rag_service = rag_service
        self._prompts_service = prompts_service
        self._pending_approval_runs: dict[str, PendingApprovalRun] = {}

    def _prune_pending_approval_runs(self) -> None:
        now = time_ns()
        expired_tokens = [
            token
            for token, pending_run in self._pending_approval_runs.items()
            if now - pending_run.created_at > PENDING_APPROVAL_TTL_NS
        ]
        for token in expired_tokens:
            self._pending_approval_runs.pop(token, None)

        if len(self._pending_approval_runs) < MAX_PENDING_APPROVAL_RUNS:
            return

        oldest_token = min(self._pending_approval_runs, key=lambda token: self._pending_approval_runs[token].created_at)
        self._pending_approval_runs.pop(oldest_token, None)

    async def _handle_user_prompt_node(self, node: UserPromptNode, history_id: UUID) -> AsyncIterator[StreamItem]:
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

    async def _handle_model_request_node(
        self,
        node: ModelRequestNode,
        run: AgentRun,
        history_id: UUID,
    ) -> AsyncIterator[StreamItem]:
        # A model request node => We can stream tokens from the model's request
        async with node.stream(run.ctx) as request_stream:
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
                                # because we at least have the separator.
                                yield current_part.add_content_and_yield_delta(content=separator + event.part.content)

                            case paim.TextPart():
                                if current_part.is_streaming_but_not_in_state(PartState.TALKING):
                                    if flushed_part := current_part.flush():
                                        yield flushed_part
                                    yield current_part.reset_to_state_and_get_part_start(PartState.TALKING)

                                elif current_part.is_not_streaming():
                                    yield current_part.reset_to_state_and_get_part_start(PartState.TALKING)

                                if event.part.has_content():
                                    yield current_part.add_content_and_yield_delta(event.part.content)

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
                            case paim.ThinkingPartDelta() | paim.TextPartDelta():
                                if event.delta.content_delta:
                                    yield current_part.add_content_and_yield_delta(content=event.delta.content_delta)
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
        node: CallToolsNode,
        run: AgentRun,
        history_id: UUID,
        skip_tool_call_ids: set[str] | None = None,
    ) -> AsyncIterator[StreamItem]:
        # A handle-response node => The model returned some data, potentially calls a tool
        skip_tool_call_ids = skip_tool_call_ids or set()
        async with node.stream(run.ctx) as handle_stream:
            async for event in handle_stream:
                if isinstance(event, paim.FunctionToolCallEvent):
                    if event.part.tool_call_id in skip_tool_call_ids:
                        continue
                    tool_call = PydanticAiMapper.map_tool_call_out(
                        pai_tool_call=event.part,
                        id=uuid4(),
                        history_id=history_id,
                    )
                    await self._history_service.add_history_item(tool_call)
                    yield tool_call
                elif isinstance(event, paim.FunctionToolResultEvent):
                    tool_result = PydanticAiMapper.map_tool_result_out(
                        pai_tool_result=event.result,
                        id=uuid4(),
                        history_id=history_id,
                    )
                    await self._history_service.add_history_item(tool_result)
                    yield tool_result

    async def _handle_end_node(self, node: EndNode, run: AgentRun, history_id: UUID) -> AsyncIterator[StreamItem]:
        yield StreamEnd(id=uuid4(), history_id=history_id, created_at=time_ns())

    async def _yield_tool_approval_requests(
        self,
        deferred_tool_requests: DeferredToolRequests,
        run: AgentRun,
        history_id: UUID,
        tool_sets: list[ToolSet],
    ) -> AsyncIterator[ToolApprovalRequest]:
        if not deferred_tool_requests.approvals:
            return

        self._prune_pending_approval_runs()

        resume_token = str(uuid4())
        approval_tool_call_ids = {tool_call.tool_call_id for tool_call in deferred_tool_requests.approvals}
        metadata = {
            tool_call_id: deferred_tool_requests.metadata[tool_call_id]
            for tool_call_id in approval_tool_call_ids
            if tool_call_id in deferred_tool_requests.metadata
        }
        self._pending_approval_runs[resume_token] = PendingApprovalRun(
            created_at=time_ns(),
            history_id=history_id,
            pai_history=list(run.all_messages()),
            tool_sets=tool_sets,
            approval_tool_call_ids=approval_tool_call_ids,
            metadata=metadata,
        )

        for tool_call in deferred_tool_requests.approvals:
            yield ToolApprovalRequest(
                id=uuid4(),
                history_id=history_id,
                created_at=time_ns(),
                resume_token=resume_token,
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                args=tool_call.args,
                metadata=metadata.get(tool_call.tool_call_id),
            )

    async def _stream_pai_run(
        self,
        pai_user_prompt: str | Sequence[paim.UserContent] | None,
        history_id: UUID,
        pai_history: Sequence[paim.ModelMessage],
        tool_sets: list[ToolSet],
        deferred_tool_results: DeferredToolResults | None = None,
        skip_tool_call_ids: set[str] | None = None,
    ) -> AsyncIterator[StreamItem]:
        pai_toolsets = [
            PydanticAIToolProvider.get_pai_toolset(tool_set, self._config.logging) for tool_set in tool_sets
        ]

        agent = Agent(model=self._llm, toolsets=pai_toolsets)
        async with agent.iter(
            pai_user_prompt,
            message_history=pai_history,
            deferred_tool_results=deferred_tool_results,
            output_type=[str, DeferredToolRequests],
        ) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node):
                    async for item in self._handle_user_prompt_node(node=node, history_id=history_id):
                        yield item
                elif Agent.is_model_request_node(node):
                    async for item in self._handle_model_request_node(node=node, run=run, history_id=history_id):
                        yield item
                elif Agent.is_call_tools_node(node):
                    async for item in self._handle_call_tools_node(
                        node=node,
                        run=run,
                        history_id=history_id,
                        skip_tool_call_ids=skip_tool_call_ids,
                    ):
                        yield item
                elif Agent.is_end_node(node):
                    async for item in self._handle_end_node(node=node, run=run, history_id=history_id):
                        yield item

            if run.result and isinstance(run.result.output, DeferredToolRequests):
                async for approval_request in self._yield_tool_approval_requests(
                    deferred_tool_requests=run.result.output,
                    run=run,
                    history_id=history_id,
                    tool_sets=tool_sets,
                ):
                    yield approval_request

    async def stream_agent_run(
        self,
        user_prompt: UserPrompt,
        last_n_history_items: int = 10,
        n_memory_items: int = 10,
        tool_sets: list[ToolSet] | None = None,
    ) -> AsyncIterator[StreamItem]:
        """
        Stream the agent run, yielding StreamItems that can be consumed by UI services.
        """
        pai_user_prompt = user_prompt.prompt
        history_id = user_prompt.history_id
        tool_sets = tool_sets or []

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

        async for item in self._stream_pai_run(
            pai_user_prompt=pai_user_prompt,
            history_id=history_id,
            pai_history=pai_history,
            tool_sets=tool_sets,
        ):
            yield item

    async def resume_agent_run(
        self,
        resume_token: str,
        approvals: list[ToolApprovalDecision],
    ) -> AsyncIterator[StreamItem]:
        self._prune_pending_approval_runs()
        pending_run = self._pending_approval_runs.pop(resume_token, None)
        if not pending_run:
            raise ValueError("No pending approval run found for the provided token.")

        approval_by_tool_call_id = {approval.tool_call_id: approval for approval in approvals}
        if set(approval_by_tool_call_id.keys()) != pending_run.approval_tool_call_ids:
            raise ValueError("Approval responses must match exactly all pending approval tool call IDs.")

        deferred_tool_results = DeferredToolResults(
            approvals={
                tool_call_id: (
                    True
                    if approval.approved
                    else ToolDenied(message=approval.denial_message or "The tool call was denied by the user.")
                )
                for tool_call_id, approval in approval_by_tool_call_id.items()
            },
            metadata=pending_run.metadata,
        )

        async for item in self._stream_pai_run(
            pai_user_prompt=None,
            history_id=pending_run.history_id,
            pai_history=pending_run.pai_history,
            tool_sets=pending_run.tool_sets,
            deferred_tool_results=deferred_tool_results,
            skip_tool_call_ids=pending_run.approval_tool_call_ids,
        ):
            yield item
