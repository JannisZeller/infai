from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID


@dataclass(frozen=True)
class LoggingConfig:
    base_path: Path
    module_logging_filename_dict: dict[str, str]
    main_logging_filename: str
    mcp_logging_filename: str


@dataclass(frozen=True)
class OpenAIConfig:
    """ "LLM config for OpenAI compatible APIs."""

    base_url: str
    api_key: str
    model_name: str
    api_type: Literal["completions", "responses"]
    openai_reasoning_effort: Literal["low", "medium", "high"] | None
    openai_reasoning_summary: Literal["concise", "detailed"] | None


@dataclass(frozen=True)
class OllamaConfig:
    """LLM config for Ollama."""

    base_url: str
    model_name: str


@dataclass(frozen=True)
class EmbedderConfig:
    """Embedder config.
    Note that this should not be changed once it is setup and the RAG collection is created.
    Otherwise there will be dimensionality mismatches between the new and existing embeddings.
    Only OpenAI compatible APIs are supported."""

    base_url: str
    api_key: str
    model_name: str
    chunk_max_chars: int
    chunk_overlap_chars: int


@dataclass(frozen=True)
class Config:
    """The Configuration for the application."""

    # UI
    ui: Literal["console"]

    # History
    history_id: UUID

    # LLM
    llm_config: OpenAIConfig | OllamaConfig

    # RAG
    qdrant_url: str | None
    embedder_config: EmbedderConfig

    # Logging
    logging: LoggingConfig
