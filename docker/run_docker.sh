#!/bin/bash

# Container configuration
IMAGE_NAME="chem277b-final"
TAG="2025.12"
CONTAINER_NAME="chem277b-session"
PORT=8888

# Workspace: parent directory of where script is located
HOST_WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

# Validate workspace path
if [ ! -d "$HOST_WORKSPACE" ]; then
  echo "[error] Workspace path does not exist: $HOST_WORKSPACE"
  exit 1
fi

echo "[run] Launching container: $CONTAINER_NAME"
echo "[info] Mounting workspace: $HOST_WORKSPACE → /workspace"

docker run -it --rm \
  --gpus all \
  --name $CONTAINER_NAME \
  -p $PORT:$PORT \
  -v "$HOST_WORKSPACE":/workspace \
  --workdir /workspace \
  --hostname chem22b_docker \
  $IMAGE_NAME:$TAG \
  bash -c "mkdir -p /workspace/.local/share/jupyter/runtime && \
           exec jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser --allow-root --NotebookApp.token=''"

