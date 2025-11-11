from src.tools.models import FunctionTool, FunctionToolSet


def create_dummy_tool_set() -> FunctionToolSet:
    def dummy_tool_function(input: str) -> str:
        return f"Dummy tool returns your input: {input}"

    return FunctionToolSet(
        name="dummy_tool_set",
        system_prompt="You are a dummy tool set that returns the input text.",
        tools=[
            FunctionTool(
                name="dummy_tool",
                function=dummy_tool_function,
                system_prompt="You are a dummy tool that returns the input text.",
            )
        ],
    )
