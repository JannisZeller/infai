import asyncio

from dotenv import load_dotenv

from src.ai.prompts import PromptsService
from src.ai.pydantic_ai.adapter import PydanticAIService
from src.ai.pydantic_ai.llm import get_openai
from src.application.chat_use_case import ChatUseCase
from src.config.factory import get_config
from src.core.database import get_engine
from src.core.logging import configure_logging
from src.history.async_sqlalchemy.adapter import AsyncSqlalchemyHistoryRepo
from src.history.service import HistoryService
from src.rag.factory import get_rag_service
from src.ui.console.adapter import ConsoleAdapter


async def main():
    config = get_config()
    configure_logging(config)

    history_repo = AsyncSqlalchemyHistoryRepo(engine=get_engine())
    history_service = HistoryService(history_repo=history_repo)

    rag_service = await get_rag_service(
        config=config,
        history_service=history_service,
    )

    ai_service = PydanticAIService(
        llm=get_openai(config),
        history_service=history_service,
        rag_service=rag_service,
        prompts_service=PromptsService(),
    )

    chat_use_case = ChatUseCase(ai_service=ai_service, history_id=config.history_id)
    console_adapter = ConsoleAdapter(chat_use_case=chat_use_case)

    await console_adapter.run()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
