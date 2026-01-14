import os
from pathlib import Path
from uuid import UUID, uuid4

from dotenv import load_dotenv

from src.config.models import Config

load_dotenv()


class InlineConfigProvider:
    """An inline implementation of the ConfigProvider."""

    @staticmethod
    def get_config() -> Config:
        return Config(
            # History
            history_id=InlineConfigProvider._get_history_id(),
            # RAG
            qdrant_url="http://localhost:6333",
            # OpenAI
            openai_base_url=InlineConfigProvider._get_openai_base_url(),
            openai_api_key=InlineConfigProvider._get_openai_api_key(),
            openai_model_name="gpt-5.2",
            openai_reasoning_effort="medium",
            openai_reasoning_summary="detailed",
            # Ollama
            ollama_base_url="http://localhost:11434/v1",
            ollama_model_name="ministral-3:3b",
            # Logging
            logging_path=Path("data/logs"),
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
    def _get_openai_api_key() -> str:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        return openai_api_key

    @staticmethod
    def _get_openai_base_url() -> str:
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if not openai_base_url:
            raise ValueError("OPENAI_BASE_URL is not set in .env")
        return openai_base_url


def get_config() -> Config:
    return InlineConfigProvider.get_config()
