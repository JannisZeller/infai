#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(dirname "$0")

exit_handler() {
    cd "$CURRENT_DIR"
}
trap exit_handler EXIT


cd "$SCRIPT_DIR/.."

# Create the storage directory if it doesn't exist
mkdir -p "$(pwd)/data/qdrant_storage"

docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
    qdrant/qdrant
