#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(dirname "$0")

exit_handler() {
    cd "$CURRENT_DIR"
}
trap exit_handler EXIT


cd "$SCRIPT_DIR"


fastmcp run server.py:mcp --transport http --port 8000
