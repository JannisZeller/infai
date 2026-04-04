import os
from typing import Any

import httpx
from fastmcp.client.auth.bearer import BearerAuth
from fastmcp.client.auth.oauth import OAuth
from key_value.aio.protocols import AsyncKeyValue
from mcp.client.session import ElicitationFnT
from pydantic_ai import FunctionToolset, Tool
from pydantic_ai.mcp import MCPServer, MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP
from pydantic_ai.toolsets import AbstractToolset

from src.config.models import LoggingConfig
from src.tools.models import FunctionToolSet, MCPBearerAuth, MCPOAuth, MCPToolSetRemote, MCPToolSetSTDIO, ToolSet


class PydanticAIToolProvider:
    @staticmethod
    def get_pai_toolset(
        tool_set: ToolSet,
        logging_config: LoggingConfig,
        elicitation_callback: ElicitationFnT | None = None,
        token_store: AsyncKeyValue | None = None,
    ) -> AbstractToolset[Any]:
        match tool_set:
            case FunctionToolSet():
                return PydanticAIToolProvider._get_function_toolset(tool_set)
            case MCPToolSetSTDIO():
                mcp_server = PydanticAIToolProvider._get_mcp_server_stdio(
                    tool_set=tool_set,
                    logging_config=logging_config,
                    elicitation_callback=elicitation_callback,
                )
                return PydanticAIToolProvider._with_mcp_wrappers(mcp_server=mcp_server, tool_set=tool_set)
            case MCPToolSetRemote():
                mcp_server = PydanticAIToolProvider._get_mcp_server_remote(
                    tool_set=tool_set,
                    elicitation_callback=elicitation_callback,
                    token_store=token_store,
                )
                return PydanticAIToolProvider._with_mcp_wrappers(mcp_server=mcp_server, tool_set=tool_set)

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
    def _with_mcp_wrappers(
        mcp_server: MCPServer,
        tool_set: MCPToolSetSTDIO | MCPToolSetRemote,
    ) -> AbstractToolset[Any]:
        return PydanticAIToolProvider._with_optional_approval_wrapper(
            mcp_server=PydanticAIToolProvider._with_tool_whitelist_wrapper(
                mcp_server=mcp_server,
                tool_set=tool_set,
            ),
            tool_set=tool_set,
        )

    @staticmethod
    def _with_tool_whitelist_wrapper(
        mcp_server: MCPServer, tool_set: MCPToolSetSTDIO | MCPToolSetRemote
    ) -> AbstractToolset[Any]:
        whitelisted_tool_names = {tool.name for tool in tool_set.tools}
        return mcp_server.filtered(filter_func=lambda _ctx, tool_def: tool_def.name in whitelisted_tool_names)

    @staticmethod
    def _with_optional_approval_wrapper(
        mcp_server: AbstractToolset[Any],
        tool_set: MCPToolSetSTDIO | MCPToolSetRemote,
    ) -> AbstractToolset[Any]:
        tool_names_requiring_approval = {tool.name for tool in tool_set.tools if tool.requires_approval}
        if not tool_names_requiring_approval:
            return mcp_server

        return mcp_server.approval_required(
            approval_required_func=lambda _ctx, tool_def, _tool_args: tool_def.name in tool_names_requiring_approval
        )

    @staticmethod
    def _get_mcp_server_stdio(
        tool_set: MCPToolSetSTDIO,
        logging_config: LoggingConfig,
        elicitation_callback: ElicitationFnT | None,
    ) -> MCPServerStdio:
        # Configure environment for the MCP subprocess
        # The subprocess output goes directly to our stdout/stderr, so we redirect it to a file

        env = {**os.environ, **(tool_set.env or {})}

        base_path = logging_config.base_path
        mcp_logging_filename = logging_config.mcp_logging_filename
        log_file = base_path / mcp_logging_filename

        env["MCP_LOG_FILE"] = str(log_file)
        env["MCP_LOG_LEVEL"] = "INFO"

        return MCPServerStdio(
            command=tool_set.command,
            args=tool_set.args,
            env=env,
            elicitation_callback=elicitation_callback,
        )

    @staticmethod
    def _get_mcp_server_remote(
        tool_set: MCPToolSetRemote,
        elicitation_callback: ElicitationFnT | None,
        token_store: AsyncKeyValue | None,
    ) -> MCPServerStreamableHTTP | MCPServerSSE:
        http_client = PydanticAIToolProvider._get_remote_http_client(tool_set=tool_set, token_store=token_store)
        match tool_set.transport:
            case "http":
                return MCPServerStreamableHTTP(
                    url=tool_set.url,
                    headers=tool_set.headers if http_client is None else None,
                    http_client=http_client,
                    elicitation_callback=elicitation_callback,
                )
            case "sse":
                return MCPServerSSE(
                    url=tool_set.url,
                    headers=tool_set.headers if http_client is None else None,
                    http_client=http_client,
                    elicitation_callback=elicitation_callback,
                )

    @staticmethod
    def _get_remote_http_client(
        tool_set: MCPToolSetRemote,
        token_store: AsyncKeyValue | None,
    ) -> httpx.AsyncClient | None:
        if tool_set.auth is None:
            return None

        match tool_set.auth:
            case MCPBearerAuth(token=token):
                auth: httpx.Auth | None = BearerAuth(token)
            case MCPOAuth(scopes=scopes, client_name=client_name, callback_port=callback_port):
                if token_store is None:
                    raise ValueError("OAuth-configured MCP servers require a configured token store.")
                auth = OAuth(
                    mcp_url=tool_set.url,
                    scopes=list(scopes),
                    client_name=client_name,
                    token_storage=token_store,
                    callback_port=callback_port,
                )

        return httpx.AsyncClient(headers=tool_set.headers or {}, auth=auth, follow_redirects=True)
