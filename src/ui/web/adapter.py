from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import cast
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mcp.client.session import ElicitationFnT

from src.ai.factory import get_ai_service
from src.ai.models import ToolApprovalDecision
from src.ai.prompts import PromptsService
from src.application.chat_use_case import ChatUseCase
from src.config.models import Config
from src.core.database import get_engine
from src.history.repo.async_sqlalchemy.adapter import AsyncSqlalchemyHistoryRepo
from src.history.service import HistoryService
from src.rag.factory import get_rag_service_or_none
from src.token_store.factory import get_token_store_or_none

# from src.tools.factories.dumcp import create_dumcp_tool_set
from src.tools.factories.dumcp_remote import create_dumcp_remote_tool_set
from src.tools.factories.dummy_tool import create_dummy_tool_set
from src.ui.web.chat_stream import serialize_stream
from src.ui.web.elicitation_service import WebMCPElicitationService
from src.ui.web.event_bus import WebEventBus
from src.ui.web.models import ChatResumeRequest, ChatTurnRequest, ElicitationResponseRequest
from src.ui.web.oauth_service import WebOAuthService
from src.ui.web.session_context import use_web_session

SESSION_COOKIE_NAME = "infai_session_id"
FRONTEND_DIST_PATH = Path(__file__).resolve().parents[3] / "frontend" / "dist"


def create_web_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = get_engine(config.database.connection_string)
        history_repo = AsyncSqlalchemyHistoryRepo(engine=engine)
        history_service = HistoryService(history_repo=history_repo)
        event_bus = WebEventBus()
        token_store = get_token_store_or_none(config=config, engine=engine)
        rag_service = await get_rag_service_or_none(config=config, history_service=history_service)
        elicitation_service = WebMCPElicitationService(event_bus=event_bus)
        oauth_service = WebOAuthService(
            event_bus=event_bus,
            public_base_url=(config.web.public_base_url if config.web else "http://localhost:8000"),
        )
        ai_service = await get_ai_service(
            config=config,
            history_service=history_service,
            rag_service=rag_service,
            prompts_service=PromptsService(),
            mcp_elicitation_callback=cast(ElicitationFnT, elicitation_service.handle_elicitation),
            mcp_token_store=token_store,
            oauth_provider_factory=oauth_service.create_auth_provider,
        )

        app.state.ai_service = ai_service
        app.state.chat_config = config.chat_config
        app.state.tool_sets = [create_dummy_tool_set(), create_dumcp_remote_tool_set()]
        app.state.event_bus = event_bus
        app.state.elicitation_service = elicitation_service
        app.state.oauth_service = oauth_service
        app.state.issued_sessions = set[str]()
        app.state.cookie_secure = bool(config.web and config.web.public_base_url.startswith("https://"))
        yield

    app = FastAPI(title="infai web", lifespan=lifespan)
    allowed_origins = list(config.web.allowed_origins) if config.web else ["http://localhost:5173"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; connect-src 'self' http: https: ws: wss:; frame-src http: https:"
        )
        return response

    @app.get("/api/session")
    async def ensure_session(response: Response, request: Request) -> dict[str, str]:
        session_id = _ensure_session_id(request=request, response=response)
        return {"session_id": session_id}

    @app.get("/api/events")
    async def event_stream(request: Request):
        session_id = _require_session_id(request=request)
        event_bus: WebEventBus = request.app.state.event_bus
        return StreamingResponse(event_bus.subscribe(session_id), media_type="text/event-stream")

    @app.post("/api/chat/turn")
    async def chat_turn(payload: ChatTurnRequest, request: Request, response: Response):
        session_id = _ensure_session_id(request=request, response=response)
        chat_use_case = _build_chat_use_case(request=request, session_id=session_id)

        async def stream_bytes():
            with use_web_session(session_id):
                stream = await chat_use_case.execute(payload.prompt)
                async for chunk in serialize_stream(stream):
                    yield chunk

        return StreamingResponse(stream_bytes(), media_type="application/x-ndjson")

    @app.post("/api/chat/resume")
    async def chat_resume(payload: ChatResumeRequest, request: Request, response: Response):
        session_id = _ensure_session_id(request=request, response=response)
        chat_use_case = _build_chat_use_case(request=request, session_id=session_id)
        approvals = [
            ToolApprovalDecision(
                tool_call_id=approval.tool_call_id,
                approved=approval.approved,
                denial_message=approval.denial_message,
            )
            for approval in payload.approvals
        ]

        async def stream_bytes():
            with use_web_session(session_id):
                stream = await chat_use_case.resume(payload.resume_token, approvals)
                async for chunk in serialize_stream(stream):
                    yield chunk

        return StreamingResponse(stream_bytes(), media_type="application/x-ndjson")

    @app.post("/api/mcp/elicitation/respond")
    async def respond_to_elicitation(payload: ElicitationResponseRequest, request: Request, response: Response):
        session_id = _require_session_id(request=request)
        try:
            await request.app.state.elicitation_service.submit_response(
                session_id=session_id,
                elicitation_id=payload.elicitation_id,
                action=payload.action,
                content=payload.content,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown elicitation id.") from exc

        return {"status": "ok"}

    @app.get("/api/mcp/oauth/callback")
    async def oauth_callback(
        request: Request,
        state: str,
        code: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
    ):
        try:
            message = await request.app.state.oauth_service.complete_callback(
                state=state,
                code=code,
                error=error,
                error_description=error_description,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown OAuth state.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return HTMLResponse(f"<html><body><p>{escape(message)}</p><script>window.close()</script></body></html>")

    if FRONTEND_DIST_PATH.exists():
        assets_path = FRONTEND_DIST_PATH / "assets"
        if assets_path.exists():
            app.mount("/assets", StaticFiles(directory=assets_path), name="frontend-assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_index(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            return FileResponse(FRONTEND_DIST_PATH / "index.html")

    else:

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_placeholder(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            return HTMLResponse(
                "<html><body><p>Run the frontend from ./frontend to use the browser UI.</p></body></html>"
            )

    return app


def _ensure_session_id(request: Request, response: Response | None = None) -> str:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    issued_sessions: set[str] = request.app.state.issued_sessions
    if session_id in issued_sessions:
        return session_id

    session_id = str(uuid4())
    issued_sessions.add(session_id)
    if response is not None:
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            httponly=True,
            samesite="lax",
            secure=request.app.state.cookie_secure,
        )
    return session_id


def _require_session_id(request: Request) -> str:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id is None or session_id not in request.app.state.issued_sessions:
        raise HTTPException(status_code=401, detail="No active browser session.")
    return session_id


def _build_chat_use_case(request: Request, session_id: str) -> ChatUseCase:
    history_id = uuid5(NAMESPACE_URL, session_id)
    return ChatUseCase(
        ai_service=request.app.state.ai_service,
        history_id=history_id,
        tool_sets=request.app.state.tool_sets,
        last_n_history_items=request.app.state.chat_config.last_n_history_items,
        n_memory_items=request.app.state.chat_config.n_memory_items,
    )
