from time import time_ns
from uuid import uuid4

from src.ai.models import StreamEnd, ToolApprovalRequest
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
