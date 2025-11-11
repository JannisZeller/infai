from typing import Protocol

from src.ai.models import SystemPrompt
from src.history.models import ModelResponse, UserPrompt


class RAGService(Protocol):
    """The Port that the RAGService expects."""

    async def search_for_user_prompt(self, user_prompt: UserPrompt, top_k: int = 10) -> SystemPrompt: ...

    async def add_history_items(self, history_items: list[UserPrompt | ModelResponse]) -> None: ...
