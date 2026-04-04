from time import time_ns
from uuid import uuid4

from src.ai.models import MCPAppLaunchRequest, StreamEnd, ToolApprovalRequest
from src.ui.console.service import ConsoleService


async def test_consume_stream_collects_tool_approval_requests():
    history_id = uuid4()
    approval_request = ToolApprovalRequest(
        id=uuid4(),
        history_id=history_id,
        created_at=time_ns(),
        resume_token="resume-token",
        tool_call_id="call_1",
        tool_name="dangerous_tool",
        args={"path": "/tmp/file"},
    )

    async def stream():
        yield approval_request
        yield StreamEnd(id=uuid4(), history_id=history_id, created_at=time_ns())

    console_service = ConsoleService()
    tool_approval_requests = await console_service.consume_stream(stream())

    assert len(tool_approval_requests) == 1
    assert tool_approval_requests[0].tool_call_id == "call_1"


async def test_consume_stream_opens_mcp_app_when_approved(monkeypatch):
    history_id = uuid4()
    opened_urls: list[str] = []
    app_launch_request = MCPAppLaunchRequest(
        id=uuid4(),
        history_id=history_id,
        created_at=time_ns(),
        tool_call_id="call_2",
        tool_name="app_tool",
        title="Example App",
        url="https://example.com/app",
    )

    async def stream():
        yield app_launch_request
        yield StreamEnd(id=uuid4(), history_id=history_id, created_at=time_ns())

    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    console_service = ConsoleService(open_browser=opened_urls.append)

    tool_approval_requests = await console_service.consume_stream(stream())

    assert tool_approval_requests == []
    assert opened_urls == ["https://example.com/app"]
