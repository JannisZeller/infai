from textwrap import dedent
from time import time_ns
from uuid import UUID, uuid4

from src.ai.models import SystemPrompt


class PromptsService:
    @staticmethod
    def main_system_prompt(history_id: UUID) -> SystemPrompt:
        return SystemPrompt(
            id=uuid4(),
            history_id=history_id,
            created_at=time_ns(),
            prompt=dedent("""
                [# General Instructions #]

                You are a helpful assistant .
            """),
        )
