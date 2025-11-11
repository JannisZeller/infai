from src.tools.models import MCPToolSetRemote


def create_dumcp_remote_tool_set() -> MCPToolSetRemote:
    return MCPToolSetRemote(
        name="dumcp_remote_tool_set",
        system_prompt="You are a dumcp remote tool set that returns the input text.",
        tools=[],
        transport="http",
        url="http://localhost:8000/mcp",
    )
