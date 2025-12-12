#!/bin/bash
# Build the Docker image using latest CUDA base

IMAGE_NAME="chem277b-final"
TAG="2025.12"

echo "[build] Building Docker image: $IMAGE_NAME:$TAG"
#docker build -t $IMAGE_NAME:$TAG .

docker build --no-cache -t $IMAGE_NAME:$TAG .

