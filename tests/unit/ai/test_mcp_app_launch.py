from time import time_ns
from uuid import uuid4

from src.ai.mcp_app_launch import parse_mcp_app_launch_request
from src.history.models import ToolResult


def test_parse_mcp_app_launch_request_from_nested_app_payload():
    tool_result = ToolResult(
        id=uuid4(),
        history_id=uuid4(),
        created_at=time_ns(),
        tool_call_id="call_1",
        tool_name="open_app",
        is_retry=False,
        result={"app": {"url": "https://example.com/app", "title": "Example App"}},
    )

    launch_request = parse_mcp_app_launch_request(tool_result)

    assert launch_request is not None
    assert launch_request.url == "https://example.com/app"
    assert launch_request.title == "Example App"


def test_parse_mcp_app_launch_request_from_json_string_payload():
    tool_result = ToolResult(
        id=uuid4(),
        history_id=uuid4(),
        created_at=time_ns(),
        tool_call_id="call_1",
        tool_name="open_app",
        is_retry=False,
        result='{"type":"mcp_app_launch","launchUrl":"https://example.com/app"}',
    )

    launch_request = parse_mcp_app_launch_request(tool_result)

    assert launch_request is not None
    assert launch_request.url == "https://example.com/app"


def test_parse_mcp_app_launch_request_ignores_non_http_payloads():
    tool_result = ToolResult(
        id=uuid4(),
        history_id=uuid4(),
        created_at=time_ns(),
        tool_call_id="call_1",
        tool_name="open_app",
        is_retry=False,
        result={"app": {"resourceUri": "ui://widget/example.html"}},
    )

    assert parse_mcp_app_launch_request(tool_result) is None
