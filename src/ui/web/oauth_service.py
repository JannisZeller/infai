import asyncio
from dataclasses import dataclass
from time import time_ns
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from fastmcp.client.auth.oauth import TokenStorageAdapter
from key_value.aio.protocols import AsyncKeyValue
from mcp.client.auth.oauth2 import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyHttpUrl, TypeAdapter

from src.token_store.session_scoped import SessionScopedAsyncKeyValue
from src.tools.models import MCPOAuth, MCPToolSetRemote
from src.ui.web.event_bus import WebEventBus
from src.ui.web.session_context import get_current_web_session_id


@dataclass
class PendingOAuthCallback:
    session_id: str
    future: asyncio.Future[tuple[str, str | None]]


class WebOAuthService:
    def __init__(self, event_bus: WebEventBus, public_base_url: str):
        self._event_bus = event_bus
        self._public_base_url = public_base_url.rstrip("/")
        self._pending_by_state: dict[str, PendingOAuthCallback] = {}

    def create_auth_provider(self, tool_set: MCPToolSetRemote, token_store: AsyncKeyValue) -> OAuthClientProvider:
        session_id = get_current_web_session_id()
        if session_id is None:
            raise ValueError("No active web session for MCP OAuth flow.")

        if not isinstance(tool_set.auth, MCPOAuth):
            raise ValueError("Web OAuth service can only be used with MCPOAuth tool sets.")

        scoped_store = SessionScopedAsyncKeyValue(key_value=token_store, session_id=session_id)
        storage = TokenStorageAdapter(async_key_value=scoped_store, server_url=tool_set.url.rstrip("/"))
        redirect_uri = f"{self._public_base_url}/api/mcp/oauth/callback"
        redirect_uri_value = TypeAdapter(AnyHttpUrl).validate_python(redirect_uri)

        client_metadata = OAuthClientMetadata(
            client_name=tool_set.auth.client_name,
            redirect_uris=[redirect_uri_value],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=" ".join(tool_set.auth.scopes) if tool_set.auth.scopes else None,
        )

        pending_state: dict[str, str] = {}

        async def redirect_handler(authorization_url: str) -> None:
            state = _parse_state(authorization_url)
            pending_state["value"] = state
            future: asyncio.Future[tuple[str, str | None]] = asyncio.get_running_loop().create_future()
            self._pending_by_state[state] = PendingOAuthCallback(session_id=session_id, future=future)
            await self._event_bus.publish(
                session_id,
                {
                    "type": "mcp_auth_required",
                    "created_at": time_ns(),
                    "authorization_url": authorization_url,
                    "server_url": tool_set.url,
                },
            )

        async def callback_handler() -> tuple[str, str | None]:
            state = pending_state.get("value")
            pending = self._pending_by_state.get(state) if state else None
            if pending is None:
                raise TimeoutError("No pending OAuth callback registered.")
            try:
                return await asyncio.wait_for(pending.future, timeout=300)
            finally:
                if state is not None:
                    self._pending_by_state.pop(state, None)
                    pending_state.pop("value", None)

        return OAuthClientProvider(
            server_url=tool_set.url.rstrip("/"),
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=cast(Any, redirect_handler),
            callback_handler=cast(Any, callback_handler),
        )

    async def complete_callback(
        self,
        state: str,
        code: str | None,
        error: str | None,
        error_description: str | None,
    ) -> str:
        pending = self._pending_by_state.pop(state, None)
        if pending is None:
            raise KeyError(state)

        if error:
            message = error_description or error
            if not pending.future.done():
                pending.future.set_exception(RuntimeError(message))
            await self._event_bus.publish(
                pending.session_id,
                {"type": "mcp_auth_failed", "created_at": time_ns(), "state": state, "message": message},
            )
            return message

        if code is None:
            message = "Missing OAuth code."
            if not pending.future.done():
                pending.future.set_exception(RuntimeError(message))
            await self._event_bus.publish(
                pending.session_id,
                {"type": "mcp_auth_failed", "created_at": time_ns(), "state": state, "message": message},
            )
            raise ValueError(message)

        if not pending.future.done():
            pending.future.set_result((code, state))

        await self._event_bus.publish(
            pending.session_id,
            {"type": "mcp_auth_completed", "created_at": time_ns(), "state": state},
        )
        return "Authentication completed. You can close this window."


def _parse_state(authorization_url: str) -> str:
    state = parse_qs(urlparse(authorization_url).query).get("state", [None])[0]
    if state is None:
        raise ValueError("OAuth authorization URL did not include a state parameter.")
    return state
