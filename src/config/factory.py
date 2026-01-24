import os
from pathlib import Path
from uuid import UUID, uuid4

from dotenv import load_dotenv

from src.config.models import Config, EmbedderConfig, LoggingConfig, OllamaConfig, OpenAIConfig
from src.core.exceptions import InvalidConfigurationError

load_dotenv()


class InlineConfigProvider:
    """An inline implementation of the ConfigProvider."""

    @staticmethod
    def get_config() -> Config:
        ollama_config = OllamaConfig(
            base_url="http://localhost:11434/v1",
            model_name="ministral-3:3b",
        )

        openai_config = OpenAIConfig(
            base_url=InlineConfigProvider._get_llm_base_url(),
            api_key=InlineConfigProvider._get_llm_api_key(),
            model_name="zai-org/GLM-4.7-Flash",  # "gpt-5.2",  # "tngtech/DeepSeek-TNG-R1T2-Chimera",  # "zai-org/GLM-4.7-Flash",
            openai_reasoning_effort="medium",
            openai_reasoning_summary="detailed",
            api_type="responses",
        )

        return Config(
            # UI
            ui="console",
            # History
            history_id=InlineConfigProvider._get_history_id(),
            # LLM
            llm_config=openai_config or ollama_config,
            # ollama_base_url="http://localhost:11434/v1",
            # ollama_model_name="ministral-3:3b",
            # RAG
            qdrant_url="http://localhost:6333",
            embedder_config=EmbedderConfig(
                base_url=InlineConfigProvider._get_embedder_base_url(),
                api_key=InlineConfigProvider._get_embedder_api_key(),
                model_name="text-embedding-3-small",
                chunk_max_chars=16000,  # Model can do 8192 tokens, i.e., we should be safe with 16k chars
                chunk_overlap_chars=1600,
            ),
            # Logging
            logging=LoggingConfig(
                base_path=Path("data/logs"),
                module_logging_filename_dict={
                    "sqlalchemy": "sqlalchemy.log",
                },
                main_logging_filename="main.log",
                mcp_logging_filename="mcp.log",
            ),
        )

    @staticmethod
    def _get_history_id() -> UUID:
        if not os.path.exists("data/history.id"):
            os.makedirs("data", exist_ok=True)
            with open("data/history.id", "w") as f:
                f.write(str(uuid4()))
        with open("data/history.id", "r") as f:
            return UUID(f.read())

    @staticmethod
    def _get_llm_base_url() -> str:
        openai_base_url = os.getenv("LLM_BASE_URL")
        if not openai_base_url:
            raise InvalidConfigurationError("LLM_BASE_URL is not set in .env")
        return openai_base_url

    @staticmethod
    def _get_llm_api_key() -> str:
        openai_api_key = os.getenv("LLM_API_KEY")
        if not openai_api_key:
            raise InvalidConfigurationError("LLM_API_KEY is not set in .env")
        return openai_api_key

    @staticmethod
    def _get_embedder_base_url() -> str:
        embedder_base_url = os.getenv("EMBEDDER_BASE_URL")
        if not embedder_base_url:
            raise InvalidConfigurationError("EMBEDDER_BASE_URL is not set in .env")
        return embedder_base_url

    @staticmethod
    def _get_embedder_api_key() -> str:
        embedder_api_key = os.getenv("EMBEDDER_API_KEY")
        if not embedder_api_key:
            raise InvalidConfigurationError("EMBEDDER_API_KEY is not set in .env")
        return embedder_api_key


def get_config() -> Config:
    return InlineConfigProvider.get_config()
