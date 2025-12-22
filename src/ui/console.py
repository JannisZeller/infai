from typing import AsyncIterator

from rich.console import Console
from rich.panel import Panel

from src.ai.models import (
    ModelResponseDelta,
    PartStart,
    StreamEnd,
    StreamItem,
    SystemPrompt,
    ThinkingDelta,
)
from src.history.service.models import ModelResponse, ThinkingStep, ToolCall, ToolResult, UserPrompt


class ConsoleService:
    """Service for rendering AI stream items to the console using Rich."""

    def __init__(self):
        self._console = Console()
        self._no_part_yet = True

    async def consume_stream(self, stream: AsyncIterator[StreamItem]):
        """
        Consume a stream of StreamItems and render them to the console.

        Args:
            stream: An async iterator of StreamItem objects from the AI service
        """
        self._no_part_yet = True

        async for item in stream:
            match item:
                case PartStart():
                    self._handle_part_start_item(item)
                case ThinkingDelta():
                    self._console.print(item.delta, end="", style="dim yellow")
                case ModelResponseDelta():
                    self._console.print(item.delta, end="", style="green")
                case ToolCall():
                    self._handle_tool_call(item)
                case ToolResult():
                    self._handle_tool_result(item)
                case StreamEnd():
                    self._console.print("\n")
                    # Don't break - let the generator complete naturally to prevent async-context issues
                case UserPrompt() | ModelResponse() | ThinkingStep() | SystemPrompt():
                    # These are already streamed in deltas and don't need to be rendered again
                    pass

    def _handle_part_start(self, label: str, style: str = "bold cyan"):
        """Handle the start of a new part with visual separation."""
        if not self._no_part_yet:
            self._console.print("\n")  # Add spacing between sections
        self._no_part_yet = False
        self._console.print(f"\n[{style}]{label}[/{style}]")

    def _handle_part_start_item(self, item: PartStart):
        """Handle different types of part starts."""
        if item.part_type == "thinking":
            self._handle_part_start("🤔 Thinking", "bold yellow")
        elif item.part_type == "response":
            self._handle_part_start("💬 Response", "bold green")
        elif item.part_type == "tool_call_prep":
            self._handle_part_start("🔧 Preparing Tool Call", "bold magenta")
        elif item.part_type == "final_response":
            self._handle_part_start("✨ Final Response", "bold bright_green")

    def _handle_tool_call(self, tool_call: ToolCall):
        """Render a tool call as a rich panel."""
        tool_panel = Panel(
            f"[cyan]Tool:[/cyan] [bold]{tool_call.tool_name}[/bold]\n"
            f"[cyan]Args:[/cyan] {tool_call.args}\n"
            f"[dim]Call ID: {tool_call.tool_call_id}[/dim]",
            title="🔧 Tool Call",
            border_style="magenta",
            padding=(0, 1),
        )
        self._console.print(tool_panel)

    def _handle_tool_result(self, tool_result: ToolResult):
        """Render a tool result as a rich panel."""
        result_panel = Panel(
            f"{tool_result.result}",
            title="✅ Tool Result",
            border_style="green",
            padding=(0, 1),
        )
        self._console.print(result_panel)
