from time import time_ns
from typing import AsyncIterator
from uuid import UUID, uuid4

from src.ai.models import StreamItem
from src.ai.service import AIService
from src.history.service.models import UserPrompt


class ChatUseCase:
    def __init__(self, ai_service: AIService, history_id: UUID):
        self._ai_service = ai_service
        self._history_id = history_id

    async def execute(self, prompt_text: str) -> AsyncIterator[StreamItem]:
        user_prompt = UserPrompt(
            id=uuid4(),
            history_id=self._history_id,
            created_at=time_ns(),
            prompt=prompt_text,
        )
        return self._ai_service.stream_agent_run(user_prompt)
