from src.tools.models import MCPTool, MCPToolSetSTDIO


def create_dumcp_tool_set() -> MCPToolSetSTDIO:
    return MCPToolSetSTDIO(
        name="dumcp_tool_set",
        system_prompt="You are a dumcp tool set for simple arithmetic operations.",
        tools=[
            MCPTool(
                name="add",
                system_prompt="Adds two integers together.",
            )
        ],
        command="python",
        args=["-m", "dumcp.server"],
        env=None,
    )
