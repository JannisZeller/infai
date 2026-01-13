import asyncio
import os
from uuid import UUID, uuid4

from dotenv import load_dotenv

from src.ai.prompts import PromptsService
from src.ai.pydantic_ai.adapter import PydanticAIService
from src.ai.pydantic_ai.llm import get_openai
from src.application.chat_use_case import ChatUseCase
from src.core.database import get_engine
from src.core.logging import configure_logging
from src.history.repo.async_sqlalchemy_repo import AsyncSqlalchemyHistoryRepo
from src.history.service.service import HistoryService
from src.rag.clients import OpenAIProvider, QdrantClientProvider
from src.rag.service import RAGService
from src.ui.console.adapter import ConsoleAdapter


def get_history_id() -> UUID:
    if not os.path.exists("data/history.id"):
        os.makedirs("data", exist_ok=True)
        with open("data/history.id", "w") as f:
            f.write(str(uuid4()))
    with open("data/history.id", "r") as f:
        return UUID(f.read())


async def main():
    configure_logging()

    history_id = get_history_id()

    qdrant_client_provider = QdrantClientProvider(qdrant_url="http://localhost:6333")
    openai_client_provider = OpenAIProvider(history_id=history_id)

    history_repo = AsyncSqlalchemyHistoryRepo(engine=get_engine())
    history_service = HistoryService(history_repo=history_repo)

    rag_service = await RAGService.create(
        qdrant_client_provider=qdrant_client_provider,
        openai_client_provider=openai_client_provider,
        history_service=history_service,
        history_id=history_id,
    )

    ai_service = PydanticAIService(
        llm=get_openai(),
        history_service=history_service,
        rag_service=rag_service,
        prompts_service=PromptsService(),
    )

    chat_use_case = ChatUseCase(ai_service=ai_service, history_id=history_id)
    console_adapter = ConsoleAdapter(chat_use_case=chat_use_case)

    await console_adapter.run()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
