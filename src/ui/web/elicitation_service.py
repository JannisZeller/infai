import asyncio
from dataclasses import dataclass
from time import time_ns
from typing import Any, Literal, cast
from uuid import uuid4

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext

from src.ui.web.event_bus import WebEventBus
from src.ui.web.session_context import get_current_web_session_id


@dataclass
class PendingElicitation:
    session_id: str
    future: asyncio.Future[mcp_types.ElicitResult]


class WebMCPElicitationService:
    def __init__(self, event_bus: WebEventBus):
        self._event_bus = event_bus
        self._pending: dict[str, PendingElicitation] = {}

    async def handle_elicitation(
        self,
        _context: RequestContext[ClientSession, Any, Any],
        params: mcp_types.ElicitRequestParams,
    ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
        session_id = get_current_web_session_id()
        if session_id is None:
            return mcp_types.ErrorData(code=mcp_types.INTERNAL_ERROR, message="No active web session for elicitation.")

        elicitation_id = str(uuid4())
        future: asyncio.Future[mcp_types.ElicitResult] = asyncio.get_running_loop().create_future()
        self._pending[elicitation_id] = PendingElicitation(session_id=session_id, future=future)

        event: dict[str, Any] = {
            "type": "mcp_elicitation_request",
            "id": elicitation_id,
            "created_at": time_ns(),
            "message": params.message,
        }

        if isinstance(params, mcp_types.ElicitRequestURLParams):
            if not params.url.startswith(("http://", "https://")):
                return mcp_types.ErrorData(
                    code=mcp_types.INVALID_PARAMS,
                    message="Only http(s) elicitation URLs are supported.",
                )
            event["kind"] = "url"
            event["url"] = params.url
        else:
            event["kind"] = "form"
            event["schema"] = params.requestedSchema

        await self._event_bus.publish(session_id, event)

        try:
            return await asyncio.wait_for(future, timeout=300)
        finally:
            self._pending.pop(elicitation_id, None)

    async def submit_response(
        self,
        session_id: str,
        elicitation_id: str,
        action: Literal["accept", "decline", "cancel"],
        content: dict[str, Any] | None,
    ) -> None:
        pending = self._pending.get(elicitation_id)
        if pending is None or pending.session_id != session_id:
            raise KeyError(elicitation_id)

        if not pending.future.done():
            pending.future.set_result(mcp_types.ElicitResult(action=cast(Any, action), content=content))
