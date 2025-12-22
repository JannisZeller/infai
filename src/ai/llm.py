import os
from functools import lru_cache

from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

# from pydantic_ai.providers.azure import AzureProvider


@lru_cache
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is not set in .env")
    provider = OpenAIProvider(
        base_url=base_url,
        api_key=api_key,
    )
    # provider = AzureProvider(
    #     azure_endpoint=os.getenv("AZURE_AIF_URL"),
    #     api_version="2025-03-01-preview",
    #     api_key=os.getenv("AZURE_AIF_KEY"),
    # )
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="detailed",
        parallel_tool_calls=True,
    )
    return OpenAIResponsesModel(
        model_name="gpt-5.2",
        provider=provider,
        settings=settings,
    )
