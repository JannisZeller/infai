from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class Tool:
    name: str
    system_prompt: str


@dataclass(frozen=True)
class FunctionTool(Tool):
    function: Callable[[Any], Any]


@dataclass(frozen=True)
class MCPTool(Tool):
    pass


@dataclass(frozen=True)
class BaseToolSet:
    name: str
    system_prompt: str


@dataclass(frozen=True)
class FunctionToolSet(BaseToolSet):
    tools: list[FunctionTool]


@dataclass(frozen=True)
class MCPToolSetSTDIO(BaseToolSet):
    tools: list[MCPTool]
    command: str
    args: list[str]
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class MCPToolSetRemote(BaseToolSet):
    tools: list[MCPTool]
    transport: Literal["http", "sse"]
    url: str


ToolSet = FunctionToolSet | MCPToolSetSTDIO | MCPToolSetRemote
