import os
from typing import Any

from pydantic_ai import FunctionToolset, Tool
from pydantic_ai.mcp import MCPServer, MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP
from pydantic_ai.toolsets import AbstractToolset

from src.config.models import LoggingConfig
from src.tools.models import FunctionToolSet, MCPToolSetRemote, MCPToolSetSTDIO, ToolSet


class PydanticAIToolProvider:
    @staticmethod
    def get_pai_toolset(tool_set: ToolSet, logging_config: LoggingConfig) -> AbstractToolset[Any]:
        match tool_set:
            case FunctionToolSet():
                return PydanticAIToolProvider._get_function_toolset(tool_set)
            case MCPToolSetSTDIO():
                mcp_server = PydanticAIToolProvider._get_mcp_server_stdio(tool_set, logging_config)
                return PydanticAIToolProvider._with_optional_approval_wrapper(mcp_server=mcp_server, tool_set=tool_set)
            case MCPToolSetRemote():
                mcp_server = PydanticAIToolProvider._get_mcp_server_remote(tool_set)
                return PydanticAIToolProvider._with_optional_approval_wrapper(mcp_server=mcp_server, tool_set=tool_set)

    @staticmethod
    def _get_function_toolset(tool_set: FunctionToolSet) -> FunctionToolset:
        tools = [
            Tool(
                function=tool.function,
                name=tool.name,
                description=tool.system_prompt,
                requires_approval=tool.requires_approval,
            )
            for tool in tool_set.tools
        ]
        return FunctionToolset(tools=tools)

    @staticmethod
    def _with_optional_approval_wrapper(
        mcp_server: MCPServer, tool_set: MCPToolSetSTDIO | MCPToolSetRemote
    ) -> AbstractToolset[Any]:
        tool_names_requiring_approval = {tool.name for tool in tool_set.tools if tool.requires_approval}
        if not tool_names_requiring_approval:
            return mcp_server

        return mcp_server.approval_required(
            approval_required_func=lambda _ctx, tool_def, _tool_args: tool_def.name in tool_names_requiring_approval
        )

    @staticmethod
    def _get_mcp_server_stdio(tool_set: MCPToolSetSTDIO, logging_config: LoggingConfig) -> MCPServerStdio:
        # Configure environment for the MCP subprocess
        # The subprocess output goes directly to our stdout/stderr, so we redirect it to a file

        env = {**os.environ, **(tool_set.env or {})}

        base_path = logging_config.base_path
        mcp_logging_filename = logging_config.mcp_logging_filename
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
