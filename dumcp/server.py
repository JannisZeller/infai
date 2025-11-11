from mcp.server.fastmcp import FastMCP

from dumcp.logging import configure_logging

mcp: FastMCP = FastMCP("Demo 🚀")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers together.

    This tool performs basic arithmetic addition of two integer values.

    Args:
        a: The first integer to add
        b: The second integer to add

    Returns:
        The sum of a and b as an integer
    """
    return a + b


if __name__ == "__main__":
    configure_logging()
    mcp.run()
