import os
import re
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import ValidationError
from yaml import YAMLError, safe_load

from src.config.models import Config, DatabaseConfig, FileConfig
from src.core.exceptions import InvalidConfigurationError

DEFAULT_CONFIG_PATH = Path("config.yaml")
CONFIG_PATH_ENV_KEY = "INFAI_CONFIG_PATH"
ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_config_path(config_path: Path | str | None = None) -> Path:
    if config_path:
        return Path(config_path)

    if env_config_path := os.getenv(CONFIG_PATH_ENV_KEY):
        return Path(env_config_path)

    return DEFAULT_CONFIG_PATH


def _load_raw_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = _resolve_config_path(config_path)
    if not path.exists():
        raise InvalidConfigurationError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = safe_load(file)
    except YAMLError as exc:
        raise InvalidConfigurationError(f"Config file is not valid YAML: {path}") from exc

    if not isinstance(loaded, dict):
        raise InvalidConfigurationError(f"Config file must contain a YAML mapping at root: {path}")

    return loaded


def _interpolate_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return _interpolate_env_string(value)

    if isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]

    if isinstance(value, dict):
        return {key: _interpolate_env_vars(item) for key, item in value.items()}

    return value


def _interpolate_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        env_key = match.group(1)
        env_value = os.getenv(env_key)
        if env_value is None:
            placeholder = f"${{env:{env_key}}}"
            raise InvalidConfigurationError(f"Missing environment variable '{env_key}' required by '{placeholder}'")
        return env_value

    return ENV_PATTERN.sub(replace, value)


def _parse_file_config(config_path: Path | str | None = None) -> FileConfig:
    path = _resolve_config_path(config_path)
    raw_config = _load_raw_config(path)
    interpolated_config = _interpolate_env_vars(raw_config)

    try:
        return FileConfig.model_validate(interpolated_config)
    except ValidationError as exc:
        raise InvalidConfigurationError(f"Invalid configuration in {path}: {exc}") from exc


def _get_history_id(path: Path) -> UUID:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            file.write(str(uuid4()))
    with path.open("r", encoding="utf-8") as file:
        return UUID(file.read().strip())


def get_database_connection_string(config_path: Path | str | None = None) -> str:
    path = _resolve_config_path(config_path)
    raw_config = _load_raw_config(path)

    raw_database = raw_config.get("database")
    if not isinstance(raw_database, dict):
        raise InvalidConfigurationError(f"Missing or invalid 'database' section in {path}")

    interpolated_database = _interpolate_env_vars(raw_database)
    try:
        return DatabaseConfig.model_validate(interpolated_database).connection_string
    except ValidationError as exc:
        raise InvalidConfigurationError(f"Invalid database configuration in {path}: {exc}") from exc


def get_config(config_path: Path | str | None = None) -> Config:
    file_config = _parse_file_config(config_path)
    llm_config = file_config.llm.openai if file_config.llm.provider == "openai" else file_config.llm.ollama

    return Config(
        ui=file_config.ui,
        history_id=_get_history_id(file_config.history.id_file_path),
        llm_config=llm_config,
        qdrant_url=file_config.rag.qdrant_url,
        embedder_config=file_config.embedder,
        logging=file_config.logging,
        chat_config=file_config.chat,
        database=file_config.database,
        token_store=file_config.token_store,
    )
