from src.tools.models import MCPOAuth, MCPTool, MCPToolSetRemote


def create_dumcp_remote_tool_set() -> MCPToolSetRemote:
    return MCPToolSetRemote(
        name="dumcp_remote_tool_set",
        system_prompt="You are a dumcp remote tool set for simple arithmetic operations.",
        tools=[
            MCPTool(
                name="dummy_tool",
                system_prompt="Dummy tool for testing.",
            ),
        ],
        transport="http",
        url="http://localhost:8002/mcp",
        auth=MCPOAuth(),
    )
