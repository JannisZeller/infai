from src.tools.models import MCPToolSetSTDIO


def create_dumcp_tool_set() -> MCPToolSetSTDIO:
    return MCPToolSetSTDIO(
        name="dumcp_tool_set",
        system_prompt="You are a dumcp tool set that returns the input text.",
        tools=[],
        command="python",
        args=["-m", "dumcp.server"],
        env=None,
    )
