from typing import Any, cast

import mcp.types as mcp_types

from src.ui.console.mcp_elicitation_service import ConsoleMCPElicitationService


async def test_handle_elicitation_form_collects_typed_values():
    responses = iter(["Ada", "42", "yes", "y"])
    output_lines: list[str] = []
    service = ConsoleMCPElicitationService(
        prompt_user=lambda _prompt: next(responses),
        print_line=output_lines.append,
    )

    params = mcp_types.ElicitRequestFormParams(
        message="Please provide profile details.",
        requestedSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "is_admin": {"type": "boolean"},
            },
            "required": ["name", "age", "is_admin"],
        },
    )

    result = await service.handle_elicitation(_context=cast(Any, None), params=params)

    assert isinstance(result, mcp_types.ElicitResult)
    assert result.action == "accept"
    assert result.content == {"name": "Ada", "age": 42, "is_admin": True}


async def test_handle_elicitation_url_declines_by_default():
    service = ConsoleMCPElicitationService(
        prompt_user=lambda _prompt: "",
        print_line=lambda _line: None,
    )

    params = mcp_types.ElicitRequestURLParams(
        message="Please complete login.",
        url="https://example.com/auth",
        elicitationId="abc-123",
    )

    result = await service.handle_elicitation(_context=cast(Any, None), params=params)

    assert isinstance(result, mcp_types.ElicitResult)
    assert result.action == "decline"
