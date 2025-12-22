import os
from functools import lru_cache

from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

# from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.azure import AzureProvider


@lru_cache
def get_llm():
    # provider = OpenAIProvider(
    #     base_url="https://taia.tngtech.com/proxy/openai/v1",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # )
    # https://zellerj-aif-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5.1/chat/completions?api-version=2025-01-01-preview
    provider = AzureProvider(
        azure_endpoint=os.getenv("AZURE_AIF_URL"),
        api_version="2025-03-01-preview",
        api_key=os.getenv("AZURE_AIF_KEY"),
    )
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="detailed",
        parallel_tool_calls=True,
    )
    return OpenAIResponsesModel(
        model_name="gpt-5.1",
        provider=provider,
        settings=settings,
    )
