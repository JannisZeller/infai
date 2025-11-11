from functools import lru_cache

from pydantic_ai import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.config.models import OllamaConfig, OpenAIConfig
from src.core.exceptions import ResourceNotAvailableError
from src.core.logging import get_logger

logger = get_logger("Startup: ", output="console", simple_format=True)


async def _ping_model(model: OpenAIResponsesModel | OpenAIChatModel):
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content="This is a ping. Respond with 'pong'.")])
    ]

    return await model.request(
        messages=messages,
        model_settings=model.settings,
        model_request_parameters=ModelRequestParameters(),
    )


async def get_openai(cfg: OpenAIConfig):
    provider = OpenAIProvider(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
    )

    settings = OpenAIResponsesModelSettings()
    settings["parallel_tool_calls"] = True

    if cfg.openai_reasoning_effort:
        settings["openai_reasoning_effort"] = cfg.openai_reasoning_effort
    if cfg.openai_reasoning_summary:
        settings["openai_reasoning_summary"] = cfg.openai_reasoning_summary

    return OpenAIResponsesModel(
        model_name=cfg.model_name,
        provider=provider,
        settings=settings,
    )


async def get_ollama(cfg: OllamaConfig):
    if not cfg.model_name:
        raise ValueError("Ollama model name is not set")

    if not cfg.base_url:
        raise ValueError("Ollama base URL is not set")

    provider = OllamaProvider(base_url=cfg.base_url)

    return OpenAIChatModel(
        model_name=cfg.model_name,
        provider=provider,
    )


@lru_cache
async def get_llm(config: OpenAIConfig | OllamaConfig) -> OpenAIResponsesModel | OpenAIChatModel:
    if isinstance(config, OpenAIConfig):
        model = await get_openai(config)
    else:
        model = await get_ollama(config)

    logger.info(f"LLM: Pinging {model.model_name}...")
    try:
        await _ping_model(model)
    except Exception as e:
        raise ResourceNotAvailableError(f"LLM: Pinging {model.model_name} failed with error: {e}")

    return model
