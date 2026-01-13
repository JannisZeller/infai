from typing import AsyncIterator, Protocol

from src.ai.models import StreamItem
from src.history.models import UserPrompt


class AIService(Protocol):
    async def stream_agent_run(
        self,
        user_prompt: UserPrompt,
        last_n_history_items: int = 10,
        n_memory_items: int = 10,
    ) -> AsyncIterator[StreamItem]:
        if False:
            yield ...  # Needed for type checking
