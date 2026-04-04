from typing import AsyncIterator, Protocol

from src.ai.models import StreamItem, ToolApprovalDecision
from src.history.models import UserPrompt
from src.tools.models import ToolSet


class AIService(Protocol):
    async def stream_agent_run(
        self,
        user_prompt: UserPrompt,
        last_n_history_items: int = 10,
        n_memory_items: int = 10,
        tool_sets: list[ToolSet] | None = None,
    ) -> AsyncIterator[StreamItem]:
        if False:
            yield ...  # Needed for type checking

    async def resume_agent_run(
        self,
        resume_token: str,
        approvals: list[ToolApprovalDecision],
    ) -> AsyncIterator[StreamItem]:
        if False:
            yield ...  # Needed for type checking
