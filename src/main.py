import asyncio

from dotenv import load_dotenv

from src.ai.factory import get_ai_service
from src.ai.prompts import PromptsService
from src.application.chat_use_case import ChatUseCase
from src.config.factory import get_config
from src.core.database import get_engine
from src.core.exceptions import InvalidConfigurationError, ResourceNotAvailableError
from src.core.logging import configure_module_logging, get_logger
from src.history.async_sqlalchemy.adapter import AsyncSqlalchemyHistoryRepo
from src.history.service import HistoryService
from src.rag.factory import get_rag_service_or_none
from src.tools.factories.dumcp import create_dumcp_tool_set  # type: ignore # noqa: F401
from src.tools.factories.dumcp_remote import create_dumcp_remote_tool_set  # type: ignore # noqa: F401
from src.tools.factories.dummy_tool import create_dummy_tool_set
from src.ui.console.adapter import ConsoleAdapter

logger = get_logger("Startup: ", output="console", simple_format=True)


async def main():
    try:
        config = get_config()
    except InvalidConfigurationError as exc:
        logger.error(exc)
        return

    configure_module_logging(config)

    history_repo = AsyncSqlalchemyHistoryRepo(engine=get_engine())
    history_service = HistoryService(history_repo=history_repo)

    try:
        rag_service = await get_rag_service_or_none(
            config=config,
            history_service=history_service,
        )
        ai_service = await get_ai_service(
            config=config,
            history_service=history_service,
            rag_service=rag_service,
            prompts_service=PromptsService(),
        )
    except ResourceNotAvailableError as exc:
        logger.error(exc)
        return

    chat_use_case = ChatUseCase(
        ai_service=ai_service,
        history_id=config.history_id,
        tool_sets=[
            create_dummy_tool_set(),
            create_dumcp_tool_set(),
            # create_dumcp_remote_tool_set(),
        ],
        last_n_history_items=config.chat_config.last_n_history_items,
        n_memory_items=config.chat_config.n_memory_items,
    )

    if config.ui == "console":
        console_adapter = ConsoleAdapter(chat_use_case=chat_use_case)
    else:
        logger.error('For now, only the "console" UI is supported.')
        return

    await console_adapter.run()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
