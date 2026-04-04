#!/usr/bin/env bash
set -euo pipefail

dotenvx run -- uv run uvicorn src.main_api:app --host 0.0.0.0 --port 8000 --reload
