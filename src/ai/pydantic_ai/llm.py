from functools import lru_cache

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.config.models import Config


@lru_cache
def get_openai(config: Config):
    provider = OpenAIProvider(
        base_url=config.openai_base_url,
        api_key=config.openai_api_key,
    )
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort=config.openai_reasoning_effort,
        openai_reasoning_summary=config.openai_reasoning_summary,
        parallel_tool_calls=True,
    )
    return OpenAIResponsesModel(
        model_name=config.openai_model_name,
        provider=provider,
        settings=settings,
    )


@lru_cache
def get_ollama(config: Config):
    provider = OllamaProvider(base_url=config.ollama_base_url)
    return OpenAIChatModel(
        model_name=config.ollama_model_name,
        provider=provider,
    )
