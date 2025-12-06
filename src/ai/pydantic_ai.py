import os
from time import time_ns
from uuid import UUID, uuid4

import pydantic_ai.messages as paim
from dotenv import load_dotenv
from pydantic_ai import Agent, AgentRun, FunctionToolset
from pydantic_ai.agent import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

# from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.azure import AzureProvider
from pydantic_graph.nodes import End as EndNode
from rich.console import Console
from rich.panel import Panel

from src.ai.mapper import PydanticAiMapper
from src.history.service.models import ModelResponse, ThinkingStep, UserPrompt
from src.history.service.service import HistoryService

load_dotenv()


def dummy_tool(string: str) -> str:
    """Repeats the input text."""
    return string


dummy_tools = FunctionToolset(tools=[dummy_tool])


def get_model():
    # provider = OpenAIProvider(
    #     base_url="https://taia.tngtech.com/proxy/openai/v1",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # )
    # https://zellerj-aif-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5.1/chat/completions?api-version=2025-01-01-preview
    provider = AzureProvider(
        azure_endpoint=os.getenv("AZURE_AIF_URL"),
        api_version="2025-03-01-preview",
        api_key=os.getenv("AZURE_AIF_KEY"),
    )
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="detailed",
        parallel_tool_calls=True,
    )
    return OpenAIResponsesModel(
        model_name="gpt-5.1",
        provider=provider,
        settings=settings,
    )


class PydanticAiAgent:
    # Streaming logic following https://ai.pydantic.dev/agents/
    def __init__(self, history_service: HistoryService, model: OpenAIResponsesModel):
        self._model = model
        self._history_service = history_service
        self._console = Console()

        self._reset_state()

    def _reset_state(self):
        self._no_part_yet = True
        self._talked = ""
        self._thought = ""
        self._toolcalling = False
        self._thinking = False
        self._talking = False

    def _handle_part_start(self, label: str, style: str = "bold cyan"):
        if not self._no_part_yet:
            self._console.print("\n")  # Add spacing between sections
        self._no_part_yet = False
        self._console.print(f"\n[{style}]{label}[/{style}]")

    async def _handle_user_prompt_node(self, node: UserPromptNode, history_id: UUID):  # type: ignore
        # print(f"[UserPrompt]\n{node.user_prompt}")
        user_prompt = PydanticAiMapper.map_user_prompt_out(
            pai_user_prompt=node.user_prompt,
            history_id=history_id,
        )
        if user_prompt:
            await self._history_service.add_history_item(user_prompt)

    async def _add_and_reset_talked(self, history_id: UUID):
        self._talking = False
        if self._talked:
            await self._history_service.add_history_item(
                ModelResponse(
                    id=uuid4(),
                    history_id=history_id,
                    created_at=time_ns(),
                    response=self._talked,
                )
            )
            self._talked = ""

    async def _add_and_reset_thought(self, history_id: UUID):
        self._thinking = False
        if self._thought:
            await self._history_service.add_history_item(
                ThinkingStep(
                    id=uuid4(),
                    history_id=history_id,
                    created_at=time_ns(),
                    thoughts=self._thought,
                )
            )
            self._thought = ""

    def _reset_toolcalling(self):
        self._toolcalling = False

    async def _handle_model_request_node(self, node: ModelRequestNode, run: AgentRun, history_id: UUID):  # type: ignore
        # A model request node => We can stream tokens from the model's request
        async with node.stream(run.ctx) as request_stream:  # type: ignore
            final_result_found = False

            async for event in request_stream:
                if isinstance(event, paim.PartStartEvent):
                    pass
                elif isinstance(event, paim.PartDeltaEvent):
                    if isinstance(event.delta, paim.TextPartDelta):
                        await self._add_and_reset_thought(history_id)
                        self._reset_toolcalling()
                        if not self._talking:
                            self._handle_part_start("💬 Response", "bold green")
                            self._talking = True
                        if event.delta.content_delta:
                            self._console.print(event.delta.content_delta, end="", style="green")
                            self._talked += event.delta.content_delta

                    elif isinstance(event.delta, paim.ThinkingPartDelta):
                        await self._add_and_reset_talked(history_id)
                        self._reset_toolcalling()
                        if not self._thinking:
                            self._handle_part_start("🤔 Thinking", "bold yellow")
                            self._thinking = True
                        if event.delta.content_delta:
                            self._console.print(event.delta.content_delta, end="", style="dim yellow")
                            self._thought += event.delta.content_delta

                    elif isinstance(event.delta, paim.ToolCallPartDelta):  # type: ignore[reportUnnecessaryComparison]
                        await self._add_and_reset_thought(history_id)
                        await self._add_and_reset_talked(history_id)
                        if not self._toolcalling:
                            self._handle_part_start("🔧 Preparing Tool Call", "bold magenta")
                            self._toolcalling = True

                elif isinstance(event, paim.FinalResultEvent):
                    final_result_found = True
                    break

            if final_result_found:
                await self._add_and_reset_thought(history_id)
                self._reset_toolcalling()
                await self._add_and_reset_talked(history_id)
                self._handle_part_start("✨ Final Response", "bold bright_green")
                async for output in request_stream.stream_text(delta=True):
                    if output:
                        self._talked += output
                        self._console.print(output, end="", style="bright_green")
                await self._add_and_reset_talked(history_id)

    async def _handle_call_tools_node(self, node: CallToolsNode, run: AgentRun, history_id: UUID):  # type: ignore
        # A handle-response node => The model returned some data, potentially calls a tool
        async with node.stream(run.ctx) as handle_stream:  # type: ignore
            async for event in handle_stream:
                if isinstance(event, paim.FunctionToolCallEvent):
                    tool_panel = Panel(
                        f"[cyan]Tool:[/cyan] [bold]{event.part.tool_name}[/bold]\n"
                        f"[cyan]Args:[/cyan] {event.part.args}\n"
                        f"[dim]Call ID: {event.part.tool_call_id}[/dim]",
                        title="🔧 Tool Call",
                        border_style="magenta",
                        padding=(0, 1),
                    )
                    self._console.print(tool_panel)
                    tool_call = PydanticAiMapper.map_tool_call_out(pai_tool_call=event.part, history_id=history_id)
                    await self._history_service.add_history_item(tool_call)
                elif isinstance(event, paim.FunctionToolResultEvent):
                    result_panel = Panel(
                        f"{event.result.content}",
                        title="✅ Tool Result",
                        border_style="green",
                        padding=(0, 1),
                    )
                    self._console.print(result_panel)
                    tool_result = PydanticAiMapper.map_tool_result_out(
                        pai_tool_result=event.result, history_id=history_id
                    )
                    await self._history_service.add_history_item(tool_result)

    async def _handle_end_node(self, node: EndNode, run: AgentRun, history_id: UUID):  # type: ignore
        pass

    async def stream_agent_run(self, user_prompt: UserPrompt):
        pai_user_prompt = user_prompt.prompt
        history_id = user_prompt.history_id
        history = await self._history_service.get_or_create_history_by_id(history_id)
        pai_history = PydanticAiMapper.map_history_in(history)

        agent = Agent(model=self._model, toolsets=[dummy_tools])
        async with agent.iter(pai_user_prompt, message_history=pai_history) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node):
                    await self._handle_user_prompt_node(node=node, history_id=history_id)  # type: ignore
                elif Agent.is_model_request_node(node):
                    await self._handle_model_request_node(node=node, run=run, history_id=history_id)  # type: ignore
                elif Agent.is_call_tools_node(node):
                    await self._handle_call_tools_node(node=node, run=run, history_id=history_id)  # type: ignore
                elif Agent.is_end_node(node):
                    await self._handle_end_node(node=node, run=run, history_id=history_id)  # type: ignore
        self._reset_state()
