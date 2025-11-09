import asyncio
import os
from pathlib import Path

from src.ai.pydantic_ai import PydanticAiAgent, get_model
from src.history.models import UserPrompt
from src.history.service import HistoryService


async def main():
    workdir = os.getcwd()
    history_path = Path(workdir) / "memory" / "history.json"
    history_service = HistoryService(path=history_path)
    pydantic_ai_agent = PydanticAiAgent(
        history_service=history_service,
        model=get_model(),
    )

    while True:
        user_prompt_str = input("Enter your prompt (q to quit): ")
        if user_prompt_str == "q":
            break
        await pydantic_ai_agent.stream_agent_run(user_prompt=UserPrompt(prompt=user_prompt_str))


if __name__ == "__main__":
    asyncio.run(main())
