from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from mcp.client.session import ElicitationFnT
from pydantic_ai import FunctionToolset
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset

from src.ai.pydantic_ai.tools import PydanticAIToolProvider
from src.config.models import LoggingConfig
from src.tools.models import FunctionTool, FunctionToolSet, MCPBearerAuth, MCPOAuth, MCPTool, MCPToolSetRemote

LOGGING_CONFIG = LoggingConfig(
    base_path=Path("data/logs"),
    module_logging_filename_dict={},
    main_logging_filename="main.log",
    mcp_logging_filename="mcp.log",
)


def test_get_pai_toolset_marks_function_tools_for_approval():
    def plain_tool(value: str) -> str:
        return value

    def risky_tool(value: str) -> str:
        return value

    tool_set = FunctionToolSet(
        name="function_tools",
        system_prompt="Function tools",
        tools=[
            FunctionTool(name="plain_tool", system_prompt="plain", function=plain_tool),
            FunctionTool(name="risky_tool", system_prompt="risky", function=risky_tool, requires_approval=True),
        ],
    )

    pai_toolset = PydanticAIToolProvider.get_pai_toolset(tool_set, LOGGING_CONFIG)

    assert isinstance(pai_toolset, FunctionToolset)
    assert pai_toolset.tools["function_tools_plain_tool"].requires_approval is False
    assert pai_toolset.tools["function_tools_risky_tool"].requires_approval is True


def test_get_pai_toolset_wraps_mcp_toolset_with_approval_required():
    tool_set = MCPToolSetRemote(
        name="mcp_tools",
        system_prompt="MCP tools",
        tools=[
            MCPTool(name="safe_mcp_tool", system_prompt="safe"),
            MCPTool(name="dangerous_mcp_tool", system_prompt="dangerous", requires_approval=True),
        ],
        transport="http",
        url="http://localhost:8000/mcp",
    )

    pai_toolset = PydanticAIToolProvider.get_pai_toolset(tool_set, LOGGING_CONFIG)

    assert isinstance(pai_toolset, ApprovalRequiredToolset)
    assert isinstance(pai_toolset.wrapped, FilteredToolset)

    allow_tool = ToolDefinition(name="mcp_tools_safe_mcp_tool")
    deny_tool = ToolDefinition(name="unknown_tool")
    assert pai_toolset.wrapped.filter_func(cast(Any, None), allow_tool) is True
    assert pai_toolset.wrapped.filter_func(cast(Any, None), deny_tool) is False


def test_get_pai_toolset_passes_elicitation_callback_to_mcp_server():
    async def elicitation_callback(_context, _params):
        raise RuntimeError("not used")

    tool_set = MCPToolSetRemote(
        name="mcp_tools",
        system_prompt="MCP tools",
        tools=[MCPTool(name="safe_mcp_tool", system_prompt="safe")],
        transport="http",
        url="http://localhost:8000/mcp",
    )

    pai_toolset = PydanticAIToolProvider.get_pai_toolset(
        tool_set=tool_set,
        logging_config=LOGGING_CONFIG,
        elicitation_callback=cast(ElicitationFnT, elicitation_callback),
    )

    assert isinstance(pai_toolset, FilteredToolset)
    assert isinstance(pai_toolset.wrapped, PrefixedToolset)
    wrapped_server = cast(MCPServerStreamableHTTP, pai_toolset.wrapped.wrapped)
    assert wrapped_server.elicitation_callback is elicitation_callback


def test_get_pai_toolset_configures_bearer_auth_for_remote_mcp_server():
    tool_set = MCPToolSetRemote(
        name="mcp_tools",
        system_prompt="MCP tools",
        tools=[MCPTool(name="safe_mcp_tool", system_prompt="safe")],
        transport="http",
        url="http://localhost:8000/mcp",
        headers={"X-Test": "1"},
        auth=MCPBearerAuth(token="secret-token"),
    )

    pai_toolset = PydanticAIToolProvider.get_pai_toolset(tool_set=tool_set, logging_config=LOGGING_CONFIG)

    assert isinstance(pai_toolset, FilteredToolset)
    assert isinstance(pai_toolset.wrapped, PrefixedToolset)
    wrapped_server = cast(MCPServerStreamableHTTP, pai_toolset.wrapped.wrapped)
    assert isinstance(wrapped_server.http_client, httpx.AsyncClient)
    assert wrapped_server.http_client.headers["X-Test"] == "1"


def test_get_pai_toolset_requires_token_store_for_oauth_remote_mcp_server():
    tool_set = MCPToolSetRemote(
        name="mcp_tools",
        system_prompt="MCP tools",
        tools=[MCPTool(name="safe_mcp_tool", system_prompt="safe")],
        transport="http",
        url="http://localhost:8000/mcp",
        auth=MCPOAuth(scopes=("openid",)),
    )

    with pytest.raises(ValueError, match="configured token store"):
        PydanticAIToolProvider.get_pai_toolset(tool_set=tool_set, logging_config=LOGGING_CONFIG)
