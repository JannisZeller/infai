from time import time_ns
from typing import AsyncIterator
from uuid import UUID, uuid4

from src.ai.models import StreamItem
from src.ai.port import AIService
from src.history.models import UserPrompt
from src.tools.models import ToolSet


class ChatUseCase:
    def __init__(self, ai_service: AIService, history_id: UUID, tool_sets: list[ToolSet]):
        self._ai_service = ai_service
        self._history_id = history_id
        self._tool_sets = tool_sets

    async def execute(self, prompt_text: str) -> AsyncIterator[StreamItem]:
        user_prompt = UserPrompt(
            id=uuid4(),
            history_id=self._history_id,
            created_at=time_ns(),
            prompt=prompt_text,
        )
        return self._ai_service.stream_agent_run(user_prompt, tool_sets=self._tool_sets)
