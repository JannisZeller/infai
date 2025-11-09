import os

import pydantic_ai.messages as paim
from dotenv import load_dotenv
from pydantic_ai import Agent, AgentRun, FunctionToolset
from pydantic_ai.agent import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_graph.nodes import End as EndNode

from src.ai.mapper import PydanticAiMapper
from src.history.models import History, ModelResponse, ThinkingStep, UserPrompt
from src.history.service import HistoryService

load_dotenv()


def dummy_tool(string: str) -> str:
    """Repeats the input text."""
    return string


dummy_tools = FunctionToolset(tools=[dummy_tool])


def get_model():
    provider = OpenAIProvider(
        base_url="https://taia.tngtech.com/proxy/openai/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="detailed",
        parallel_tool_calls=True,
    )
    return OpenAIResponsesModel(
        model_name="gpt-5-mini",
        provider=provider,
        settings=settings,
    )


class PydanticAiAgent:
    # Streaming logic following https://ai.pydantic.dev/agents/
    def __init__(self, history_service: HistoryService, model: OpenAIResponsesModel):
        self._model = model
        self._history_service = history_service

        self._reset_state()

    def _reset_state(self):
        self._no_part_yet = True
        self._talked = ""
        self._thought = ""
        self._toolcalling = False
        self._thinking = False
        self._talking = False

    def _handle_part_start(self, start: str):
        if self._no_part_yet:
            self._no_part_yet = False
            start = "\n" + start
        else:
            start = "\n\n" + start
        print(start)

    async def _handle_user_prompt_node(self, node: UserPromptNode):  # type: ignore
        # print(f"[UserPrompt]\n{node.user_prompt}")
        user_prompt = PydanticAiMapper.map_user_prompt_out(node.user_prompt)
        if user_prompt:
            self._history_service.add_history_items([user_prompt])

    def _add_and_reset_talked(self):
        self._talking = False
        if self._talked:
            self._history_service.add_history_items([ModelResponse(response=self._talked)])
            self._talked = ""

    def _add_and_reset_thought(self):
        self._thinking = False
        if self._thought:
            self._history_service.add_history_items([ThinkingStep(thoughts=self._thought)])
            self._thought = ""

    def _reset_toolcalling(self):
        self._toolcalling = False

    async def _handle_model_request_node(self, node: ModelRequestNode, run: AgentRun):  # type: ignore
        # A model request node => We can stream tokens from the model's request
        async with node.stream(run.ctx) as request_stream:  # type: ignore
            final_result_found = False

            async for event in request_stream:
                if isinstance(event, paim.PartStartEvent):
                    pass
                elif isinstance(event, paim.PartDeltaEvent):
                    if isinstance(event.delta, paim.TextPartDelta):
                        self._add_and_reset_thought()
                        self._reset_toolcalling()
                        if not self._talking:
                            self._handle_part_start("[Talking - TextPart]")
                            self._talking = True
                        print(event.delta.content_delta, end="")
                        self._talked += event.delta.content_delta

                    elif isinstance(event.delta, paim.ThinkingPartDelta):
                        self._add_and_reset_talked()
                        self._reset_toolcalling()
                        if not self._thinking:
                            self._handle_part_start("[Thinking]")
                            self._thinking = True
                        print(event.delta.content_delta, end="")
                        self._thought += event.delta.content_delta or ""

                    elif isinstance(event.delta, paim.ToolCallPartDelta):  # type: ignore[reportUnnecessaryComparison]
                        self._add_and_reset_thought()
                        self._add_and_reset_talked()
                        if not self._toolcalling:
                            self._handle_part_start("[Preparing ToolCall]")
                            self._toolcalling = True

                elif isinstance(event, paim.FinalResultEvent):
                    final_result_found = True
                    break

            if final_result_found:
                self._add_and_reset_thought()
                self._reset_toolcalling()
                self._add_and_reset_talked()
                self._handle_part_start("[Talking - finally]")
                async for output in request_stream.stream_text(delta=True):
                    self._talked += output
                    print(output)
                self._add_and_reset_talked()

    async def _handle_call_tools_node(self, node: CallToolsNode, run: AgentRun):  # type: ignore
        # A handle-response node => The model returned some data, potentially calls a tool
        async with node.stream(run.ctx) as handle_stream:  # type: ignore
            async for event in handle_stream:
                if isinstance(event, paim.FunctionToolCallEvent):
                    print(
                        f" -The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})"
                    )
                    tool_call = PydanticAiMapper.map_tool_call_out(event.part)
                    self._history_service.add_history_items([tool_call])
                elif isinstance(event, paim.FunctionToolResultEvent):
                    print(f" - Tool call {event.tool_call_id!r} returned => {event.result.content}")
                    tool_result = PydanticAiMapper.map_tool_result_out(event.result)
                    self._history_service.add_history_items([tool_result])

    async def _handle_end_node(self, node: EndNode, run: AgentRun):  # type: ignore
        pass

    async def stream_agent_run(self, user_prompt: UserPrompt):
        pai_user_prompt = user_prompt.prompt
        history = self._history_service.history
        pai_history = PydanticAiMapper.map_history_in(history or History(items=[]))
        agent = Agent(model=self._model, toolsets=[dummy_tools])
        async with agent.iter(pai_user_prompt, message_history=pai_history) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node):
                    await self._handle_user_prompt_node(node)  # type: ignore
                elif Agent.is_model_request_node(node):
                    await self._handle_model_request_node(node, run)  # type: ignore
                elif Agent.is_call_tools_node(node):
                    await self._handle_call_tools_node(node, run)  # type: ignore
                elif Agent.is_end_node(node):
                    await self._handle_end_node(node, run)  # type: ignore
        self._reset_state()
