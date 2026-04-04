from time import time_ns
from uuid import uuid4

from src.ai.models import MCPAppLaunchRequest, StreamEnd
from src.ui.web.chat_stream import serialize_stream


async def test_serialize_stream_uses_expected_item_type_names():
    history_id = uuid4()

    async def stream():
        yield MCPAppLaunchRequest(
            id=uuid4(),
            history_id=history_id,
            created_at=time_ns(),
            tool_call_id="call_1",
            tool_name="app_tool",
            url="https://example.com/app",
        )
        yield StreamEnd(id=uuid4(), history_id=history_id, created_at=time_ns())

    chunks = [chunk async for chunk in serialize_stream(stream())]

    assert b'"type": "mcp_app_launch_request"' in chunks[0]
    assert b'"type": "stream_end"' in chunks[1]
