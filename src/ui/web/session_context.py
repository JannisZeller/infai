from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_CURRENT_WEB_SESSION_ID: ContextVar[str | None] = ContextVar("current_web_session_id", default=None)


def get_current_web_session_id() -> str | None:
    return _CURRENT_WEB_SESSION_ID.get()


@contextmanager
def use_web_session(session_id: str) -> Iterator[None]:
    token: Token[str | None] = _CURRENT_WEB_SESSION_ID.set(session_id)
    try:
        yield
    finally:
        _CURRENT_WEB_SESSION_ID.reset(token)
