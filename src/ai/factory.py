from key_value.aio.protocols import AsyncKeyValue
from mcp.client.session import ElicitationFnT

from src.ai.port import AIService
from src.ai.prompts import PromptsService
from src.ai.pydantic_ai.adapter import PydanticAIService
from src.ai.pydantic_ai.llm import get_llm
from src.config.models import Config
from src.history.service import HistoryService
from src.rag.port import RAGService


async def get_ai_service(
    config: Config,
    history_service: HistoryService,
    rag_service: RAGService | None,
    prompts_service: PromptsService,
    mcp_elicitation_callback: ElicitationFnT | None = None,
    mcp_token_store: AsyncKeyValue | None = None,
) -> AIService:
    llm = await get_llm(config.llm_config)

    return PydanticAIService(
        config=config,
        llm=llm,
        history_service=history_service,
        rag_service=rag_service,
        prompts_service=prompts_service,
        mcp_elicitation_callback=mcp_elicitation_callback,
        mcp_token_store=mcp_token_store,
    )
