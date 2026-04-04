import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

MAX_PENDING_EVENTS_PER_SESSION = 100


class WebEventBus:
    def __init__(self):
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}

    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        queue = self._queue(session_id)
        if queue.full():
            queue.get_nowait()
        await queue.put(event)

    async def subscribe(self, session_id: str) -> AsyncIterator[str]:
        queue = self._queue(session_id)
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            self._queues.pop(session_id, None)

    def _queue(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        if session_id not in self._queues:
            self._queues[session_id] = asyncio.Queue(maxsize=MAX_PENDING_EVENTS_PER_SESSION)
        return self._queues[session_id]
