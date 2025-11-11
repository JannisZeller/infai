from pydantic_ai import FunctionToolset
from pydantic_ai.mcp import MCPServer, MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP

from src.config.factory import get_config
from src.tools.models import FunctionToolSet, MCPToolSetRemote, MCPToolSetSTDIO, ToolSet


class PydanticAIToolProvider:
    @staticmethod
    def get_pai_toolset(tool_set: ToolSet) -> FunctionToolset | MCPServer:
        match tool_set:
            case FunctionToolSet():
                return FunctionToolset(tools=[tool.function for tool in tool_set.tools])
            case MCPToolSetSTDIO():
                return PydanticAIToolProvider._get_mcp_server_stdio(tool_set)
            case MCPToolSetRemote():
                return PydanticAIToolProvider._get_mcp_server_remote(tool_set)

    @staticmethod
    def _get_mcp_server_stdio(tool_set: MCPToolSetSTDIO) -> MCPServerStdio:
        # Configure environment for the MCP subprocess
        # The subprocess output goes directly to our stdout/stderr, so we redirect it to a file

        config = get_config()

        env = tool_set.env or {}

        base_path = config.logging.base_path
        mcp_logging_filename = config.logging.mcp_logging_filename
        log_file = base_path / mcp_logging_filename

        env["MCP_LOG_FILE"] = str(log_file)
        env["MCP_LOG_LEVEL"] = "INFO"

        return MCPServerStdio(command=tool_set.command, args=tool_set.args, env=env)

    @staticmethod
    def _get_mcp_server_remote(tool_set: MCPToolSetRemote) -> MCPServerStreamableHTTP | MCPServerSSE:
        match tool_set.transport:
            case "http":
                return MCPServerStreamableHTTP(url=tool_set.url)
            case "sse":
                return MCPServerSSE(url=tool_set.url)
