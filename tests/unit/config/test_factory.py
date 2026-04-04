from pathlib import Path
from uuid import UUID

import pytest

from src.config.factory import get_config, get_database_connection_string
from src.config.models import OpenAIConfig
from src.core.exceptions import InvalidConfigurationError


def _write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_get_config_loads_yaml_and_interpolates_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDER_BASE_URL", "https://embedder.example.test/v1")
    monkeypatch.setenv("EMBEDDER_API_KEY", "embedder-key")

    history_file = tmp_path / "history.id"
    config_path = _write_config(
        tmp_path,
        f"""
ui: console
history:
  id_file_path: {history_file}
llm:
  provider: openai
  openai:
    base_url: ${{env:LLM_BASE_URL}}
    api_key: ${{env:LLM_API_KEY}}
    model_name: gpt-5.2
    openai_reasoning_effort: medium
    openai_reasoning_summary: detailed
  ollama:
    base_url: http://localhost:11434/v1
    model_name: ministral-3:3b
rag:
  qdrant_url: http://localhost:6333
embedder:
  base_url: ${{env:EMBEDDER_BASE_URL}}
  api_key: ${{env:EMBEDDER_API_KEY}}
  model_name: text-embedding-3-small
  chunk_max_chars: 16000
  chunk_overlap_chars: 1600
logging:
  base_path: data/logs
  module_logging_filename_dict:
    sqlalchemy: sqlalchemy.log
  main_logging_filename: main.log
  mcp_logging_filename: mcp.log
chat:
  last_n_history_items: 10
  n_memory_items: 10
database:
  connection_string: sqlite+aiosqlite:///data/database.db
""",
    )

    config = get_config(config_path)

    assert isinstance(config.llm_config, OpenAIConfig)
    assert config.llm_config.base_url == "https://llm.example.test/v1"
    assert config.llm_config.api_key == "llm-key"
    assert config.embedder_config.base_url == "https://embedder.example.test/v1"
    assert history_file.exists()
    assert config.history_id == UUID(history_file.read_text(encoding="utf-8").strip())


def test_get_config_raises_on_missing_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)

    config_path = _write_config(
        tmp_path,
        """
ui: console
history:
  id_file_path: data/history.id
llm:
  provider: openai
  openai:
    base_url: ${env:MISSING_KEY}
    api_key: key
    model_name: gpt-5.2
    openai_reasoning_effort: medium
    openai_reasoning_summary: detailed
  ollama:
    base_url: http://localhost:11434/v1
    model_name: ministral-3:3b
rag:
  qdrant_url: null
embedder:
  base_url: http://localhost:1234/v1
  api_key: key
  model_name: text-embedding-3-small
  chunk_max_chars: 16000
  chunk_overlap_chars: 1600
logging:
  base_path: data/logs
  module_logging_filename_dict:
    sqlalchemy: sqlalchemy.log
  main_logging_filename: main.log
  mcp_logging_filename: mcp.log
chat:
  last_n_history_items: 10
  n_memory_items: 10
database:
  connection_string: sqlite+aiosqlite:///data/database.db
""",
    )

    with pytest.raises(InvalidConfigurationError, match="MISSING_KEY"):
        get_config(config_path)


def test_get_database_connection_string_reads_only_database_section(tmp_path: Path):
    database_connection_string = "sqlite+aiosqlite:///tmp/smoke.db"
    config_path = _write_config(
        tmp_path,
        f"""
database:
  connection_string: {database_connection_string}
""",
    )

    assert get_database_connection_string(config_path) == database_connection_string
