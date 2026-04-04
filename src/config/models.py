from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LoggingConfig(FrozenModel):
    base_path: Path
    module_logging_filename_dict: dict[str, str]
    main_logging_filename: str
    mcp_logging_filename: str


class OpenAIConfig(FrozenModel):
    """ "LLM config for OpenAI compatible APIs."""

    base_url: str
    api_key: str
    model_name: str
    openai_reasoning_effort: Literal["low", "medium", "high"] | None
    openai_reasoning_summary: Literal["concise", "detailed"] | None


class OllamaConfig(FrozenModel):
    """LLM config for Ollama."""

    base_url: str
    model_name: str


class EmbedderConfig(FrozenModel):
    """Embedder config.
    Note that this should not be changed once it is setup and the RAG collection is created.
    Otherwise there will be dimensionality mismatches between the new and existing embeddings.
    Only OpenAI compatible APIs are supported."""

    base_url: str
    api_key: str
    model_name: str
    chunk_max_chars: int
    chunk_overlap_chars: int


class ChatConfig(FrozenModel):
    """Chat config."""

    last_n_history_items: int  # The number of history items to use for each chat iteration
    n_memory_items: int  # The number of memory items to use for each chat iteration


class DatabaseConfig(FrozenModel):
    connection_string: str


class TokenStoreConfig(FrozenModel):
    encryption_key: str
    default_collection: str = "default"


class HistoryConfig(FrozenModel):
    id_file_path: Path


class RAGConfig(FrozenModel):
    qdrant_url: str | None


class LLMProviderConfig(FrozenModel):
    provider: Literal["openai", "ollama"]
    openai: OpenAIConfig
    ollama: OllamaConfig


class FileConfig(FrozenModel):
    ui: Literal["console"]
    history: HistoryConfig
    llm: LLMProviderConfig
    rag: RAGConfig
    embedder: EmbedderConfig
    logging: LoggingConfig
    chat: ChatConfig
    database: DatabaseConfig
    token_store: TokenStoreConfig | None = None


class Config(FrozenModel):
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

    # Chat
    chat_config: ChatConfig

    # Database
    database: DatabaseConfig

    # Token Store
    token_store: TokenStoreConfig | None
