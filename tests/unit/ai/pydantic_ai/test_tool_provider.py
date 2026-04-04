from pathlib import Path

from pydantic_ai import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset

from src.ai.pydantic_ai.tools import PydanticAIToolProvider
from src.config.models import LoggingConfig
from src.tools.models import FunctionTool, FunctionToolSet, MCPTool, MCPToolSetRemote

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
    assert pai_toolset.tools["plain_tool"].requires_approval is False
    assert pai_toolset.tools["risky_tool"].requires_approval is True


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
