# Makefile for Docker image and container management

IMAGE_NAME = chem277b-final
TAG        = 2025.12
CONTAINER  = chem277b-session
PORT       = 8888

.PHONY: build run clean

# Build the Docker image
build:
	@echo "[build] Building Docker image: $(IMAGE_NAME):$(TAG)"
	docker build -t $(IMAGE_NAME):$(TAG) -f Dockerfile .

# Force rebuild (no cache)
rebuild:
	@echo "[build] Building Docker image: $(IMAGE_NAME):$(TAG)"
	docker build --no-cache -t $(IMAGE_NAME):$(TAG) -f Dockerfile .

# Run the container (Jupyter Lab on port 8888)
run:
	@echo "[run] Launching container: $(CONTAINER)"
	@HOST_WORKSPACE=$$(cd .. && pwd); \
    if [ ! -d $$HOST_WORKSPACE ]; then \
      echo "[error] Workspace path does not exist: $$HOST_WORKSPACE"; \
      exit 1; \
    fi; \
    echo "[info] Mounting workspace: $$HOST_WORKSPACE -> /workspace"; \
    docker run -it --rm \
      --name $(CONTAINER) \
      -p $(PORT):$(PORT) \
      -v $$HOST_WORKSPACE:/workspace \
      --workdir /workspace \
      --hostname chem22b_docker \
      $(IMAGE_NAME):$(TAG) \
      bash -c "mkdir -p /workspace/.local/share/jupyter/runtime && \
               exec jupyter lab --ip=0.0.0.0 --port=$(PORT) --no-browser --allow-root --NotebookApp.token=''"


# Run the container (w/gpu Jupyter Lab on port 8888)
run:
	@echo "[run] Launching container: $(CONTAINER)"
	@HOST_WORKSPACE=$$(cd .. && pwd); \
    if [ ! -d $$HOST_WORKSPACE ]; then \
      echo "[error] Workspace path does not exist: $$HOST_WORKSPACE"; \
      exit 1; \
    fi; \
    echo "[info] Mounting workspace: $$HOST_WORKSPACE -> /workspace"; \
    docker run -it --rm \
      --gpus all \
      --name $(CONTAINER) \
      -p $(PORT):$(PORT) \
      -v $$HOST_WORKSPACE:/workspace \
      --workdir /workspace \
      --hostname chem22b_docker \
      $(IMAGE_NAME):$(TAG) \
      bash -c "mkdir -p /workspace/.local/share/jupyter/runtime && \
               exec jupyter lab --ip=0.0.0.0 --port=$(PORT) --no-browser --allow-root --NotebookApp.token=''"

# Clean up Docker resources
clean:
	@echo "[clean] Removing container: $(CONTAINER)"
	@docker rm -f $(CONTAINER) 2>/dev/null || true
	@echo "[clean] Removing image: $(IMAGE_NAME):$(TAG)"
	@docker rmi $(IMAGE_NAME):$(TAG) 2>/dev/null || true
	@echo "[clean] Pruning unused Docker resources..."
	@docker system prune -f

