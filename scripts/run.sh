#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(dirname "$0")

exit_handler() {
    cd "$CURRENT_DIR"
}
trap exit_handler EXIT


cd "$SCRIPT_DIR/.."

set -a
PYTHONPATH=.
set +a

dotenvx run -- ./.venv/bin/python src/main.py
