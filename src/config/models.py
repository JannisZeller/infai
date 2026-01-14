from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID


@dataclass(frozen=True)
class Config:
    """The Configuration for the application."""

    # History
    history_id: UUID

    # RAG
    qdrant_url: str

    # OpenAI
    openai_base_url: str
    openai_api_key: str
    openai_model_name: str
    openai_reasoning_effort: Literal["low", "medium", "high"]
    openai_reasoning_summary: Literal["concise", "detailed"]

    # Ollama
    ollama_base_url: str
    ollama_model_name: str

    # Logging
    logging_path: Path
