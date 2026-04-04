# infai

This repository contains a python based AI tool focused on a single session-setup for a user that
retains memory by the plain chat history (./src/history/) as well as a RAG system (./src/rag/) based
on qdrant.

## Architecture

We follow a hexagonal architecture. Every service or sub-service (e.g. repository) must implement
a suitable "port" that defines the public interface. The implementation for a specific technology
should then be in an "adapter".

The idiomatic folder structure therefore is (roughly):

```text
src/
└─ <feature-name>/
  ├─ models.py  # domain models (dataclasses or pydantic BaseModels)
  ├─ port.py    # Protocols for public interfaces of (sub)services / service
  └─ <actual-implementation>/
    ├─ models.py   # optional: implementation-specific models
    ├─ mapper.py   # optional: domain ↔ implementation mapping
    └─ adapter.py  # port implementation
```

_Notes:_
- Not the whole codebase already follows this pattern, but the history repository (./src/history/repo/)
  is a good example on how this should look.
- The ./src/core is an exception from this pattern.

## Repository layout (current)

High-level map of what lives where (beyond the generic hexagonal sketch above):

- **src/application/** — use-case orchestration (e.g. chat flow).
- **src/ai/** — LLM-facing port and **pydantic_ai** adapter (`pydantic_ai/`), prompts, AI models.
  The AI heavy lifting is done by https://ai.pydantic.dev/.
- **src/config/** — frozen dataclass config and a factory that builds it (typically from env).
- **src/core/** — shared infrastructure: async DB engine, logging, exceptions (not feature-hexagonal).
- **src/history/** — history domain models, **service**, and **repo** port with async SQLAlchemy adapter.
- **src/rag/** — RAG port and **qdrant** implementation (embeddings / vector store).
- **src/tools/** — tool definitions and factories (including optional MCP-related wiring).
- **src/ui/** — UI port and **console** adapter.
- **src/main.py** — async entrypoint; wires factories and starts the UI.
- **dumcp/** — small standalone stdio MCP server used for local tool experiments (optional).

Local state (Qdrant files, logs, etc.) lives under **data/** and is gitignored.

## Tooling and quality

- **Python** `>=3.12` (see `pyproject.toml`).
- **uv** for environments and commands: `uv sync`, `uv run pytest`, `uv run ty check`.
- **Ruff** — formatter + lint; line length **120** (`[tool.ruff]` in `pyproject.toml`).
- **ty** — static type checking; project rules are strict (`[tool.ty.rules]`).
- **pre-commit** — YAML checks, whitespace, Ruff (imports / `F401` + format), `ty`, full `pytest` (see `.pre-commit-config.yaml`).

## Running

- **App:** `./scripts/run.sh` — sets `PYTHONPATH=.` and runs `dotenvx run -- … python src/main.py` so `.env` is loaded and `from src.…` imports resolve.
- **Qdrant (local):** `./scripts/run-qudrant.sh` when you need a vector DB beside the app.

## Testing

- Unit tests are placed in ./tests/unit/.
- Integration tests are placed in ./tests/integration/.
- Shared test helpers (comparators, builders, etc.) go under ./tests/utils/ when they are reused.
- Tests are run async-native with pytest using `uv run pytest`.
- Reusable fixtures are set up directly in ./tests/conftest.py.
- As long as possible we use MagicMock or AsyncMock for mocking dependencies.
  For typing we use the `as_mock` and `as_async_mock` helpers from ./tests/conftest.py
- We want to avoid monkey-patching for testing.

## Style

- We do not want comments spammed all over the codebase.
- Comments must only exist if they explain something significant.
- In long files "heading-ish" comments of the style
  ```python
  #
  # <heading>
  #
  ```
  might exist.
- We ONLY use absolute imports (`from src.…`); the project root is on the path for tests (`pythonpath` in `pyproject.toml`) and for runs via `PYTHONPATH=.` as in `scripts/run.sh`.
