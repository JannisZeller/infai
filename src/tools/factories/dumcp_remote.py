from src.tools.models import MCPTool, MCPToolSetRemote


def create_dumcp_remote_tool_set() -> MCPToolSetRemote:
    return MCPToolSetRemote(
        name="dumcp_remote_tool_set",
        system_prompt="You are a dumcp remote tool set for simple arithmetic operations.",
        tools=[
            MCPTool(
                name="add",
                system_prompt="Adds two integers together.",
            )
        ],
        transport="http",
        url="http://localhost:8000/mcp",
    )
