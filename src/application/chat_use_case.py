from time import time_ns
from typing import AsyncIterator
from uuid import UUID, uuid4

from src.ai.models import StreamItem
from src.ai.port import AIService
from src.history.models import UserPrompt
from src.tools.models import ToolSet


class ChatUseCase:
    def __init__(
        self,
        ai_service: AIService,
        history_id: UUID,
        tool_sets: list[ToolSet],
        last_n_history_items: int = 10,
        n_memory_items: int = 10,
    ):
        self._ai_service = ai_service
        self._history_id = history_id
        self._tool_sets = tool_sets
        self._last_n_history_items = last_n_history_items
        self._n_memory_items = n_memory_items

    async def execute(self, prompt_text: str) -> AsyncIterator[StreamItem]:
        user_prompt = UserPrompt(
            id=uuid4(),
            history_id=self._history_id,
            created_at=time_ns(),
            prompt=prompt_text,
        )
        return self._ai_service.stream_agent_run(
            user_prompt,
            last_n_history_items=self._last_n_history_items,
            n_memory_items=self._n_memory_items,
            tool_sets=self._tool_sets,
        )
