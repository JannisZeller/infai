import asyncio
import os
from time import time_ns
from uuid import UUID, uuid4

from src.ai.service import AIService, get_model
from src.core.database import get_engine
from src.core.logging import configure_logging
from src.history.repo.repo import HistoryRepo
from src.history.service.models import UserPrompt
from src.history.service.service import HistoryService
from src.rag.clients import AzureOpenAIClientProvider, QdrantClientProvider
from src.rag.service import RAGService
from src.ui.console import ConsoleService


def get_history_id() -> UUID:
    if not os.path.exists("data/history.id"):
        with open("data/history.id", "w") as f:
            f.write(str(uuid4()))
    with open("data/history.id", "r") as f:
        return UUID(f.read())


async def main():
    configure_logging()

    history_id = get_history_id()

    qdrant_client_provider = QdrantClientProvider(qdrant_url="http://localhost:6333")
    azure_openai_client_provider = AzureOpenAIClientProvider(history_id=history_id)

    history_repo = HistoryRepo(engine=get_engine())
    history_service = HistoryService(history_repo=history_repo)

    rag_service = await RAGService.create(
        qdrant_client_provider=qdrant_client_provider,
        azure_openai_client_provider=azure_openai_client_provider,
        history_service=history_service,
        history_id=history_id,
    )

    pydantic_ai_agent = AIService(
        history_service=history_service,
        model=get_model(),
        rag_service=rag_service,
    )

    console_ui = ConsoleService()

    while True:
        user_prompt_str = input('\n\n ❯ Enter your prompt ("q" to quit): ')
        if user_prompt_str == "q":
            break

        stream = pydantic_ai_agent.stream_agent_run(
            user_prompt=UserPrompt(
                id=uuid4(),
                history_id=history_id,
                created_at=time_ns(),
                prompt=user_prompt_str,
            ),
        )
        await console_ui.consume_stream(stream)


if __name__ == "__main__":
    asyncio.run(main())
