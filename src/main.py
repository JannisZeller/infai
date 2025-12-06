import asyncio
import os
from time import time_ns
from uuid import UUID, uuid4

from src.ai.pydantic_ai import PydanticAiAgent, get_model
from src.core.database import get_engine
from src.core.logging import configure_logging
from src.history.repo.repo import HistoryRepo
from src.history.service.models import UserPrompt
from src.history.service.service import HistoryService


def get_history_id() -> UUID:
    if not os.path.exists("data/history.id"):
        with open("data/history.id", "w") as f:
            f.write(str(uuid4()))
    with open("data/history.id", "r") as f:
        return UUID(f.read())


async def main():
    configure_logging()

    history_id = get_history_id()

    history_repo = HistoryRepo(engine=get_engine())
    history_service = HistoryService(repo=history_repo)
    pydantic_ai_agent = PydanticAiAgent(
        history_service=history_service,
        model=get_model(),
    )

    while True:
        user_prompt_str = input('\n\nEnter your prompt ("q" to quit): ')
        if user_prompt_str == "q":
            break
        await pydantic_ai_agent.stream_agent_run(
            user_prompt=UserPrompt(
                id=uuid4(),
                history_id=history_id,
                created_at=time_ns(),
                prompt=user_prompt_str,
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
