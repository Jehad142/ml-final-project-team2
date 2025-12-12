# ml-final-project-team2

Final project for MSSE Machine Learning course.  
This repository contains src, data, and Docker configuration for building reproducible ML pipelines.

---

## Setup, Build, Run, and Clean Instructions

Prerequisites:  
You will need Linux or Windows Subsystem for Linux (WSL2 recommended) and Docker installed and running. On Ubuntu/Debian you can install Docker with:

```bash
sudo apt update
sudo apt install docker.io
sudo systemctl enable --now docker
```

On WSL, install Docker Desktop for Windows and enable WSL integration. Verify installation with:

```bash
docker --version
```

Setup Docker user
```
sudo groupadd docker        # creates the group if it doesn’t exist
sudo usermod -aG docker $USER
```

Then log out and back in (or run `newgrp docker`) so the group membership takes effect. After that you can run:
```
docker run hello-world
```

To build the image, from the project root (`ml-final-project-team2`) run:

```bash
make build
```

This delegates into `docker/Makefile` and builds the image defined in `docker/Dockerfile`. The image includes CUDA 12.8 runtime, JupyterLab (Notebook 7), and Python dependencies from `docker/requirements.txt`.

To run the container and launch JupyterLab:

```bash
make run
```

This will start the container with GPU support (`--gpus all`), mount your project root into `/workspace` inside the container, expose JupyterLab on port `8888`, disable authentication for local development (`--ServerApp.token=''`), and start JupyterLab at `http://127.0.0.1:8888/lab`. Stop the container with `Ctrl+C`.

To clean up stopped containers and dangling images:

```bash
make clean
```

Project layout:  
- `src/`: ML pipeline code and notebooks for `jarvis` and `nomad` datasets  
- `data/`: Raw and preprocessed datasets  
- `docker/`: Container build files (`Dockerfile`, `requirements.txt`, helper scripts)  
- `docs/`: Project proposal, ideas, and user guide  

Quick test:  
After `make run`, open your browser at:

```
http://127.0.0.1:8888/lab
```

Navigate to `/workspace/src/jarvis/notebooks/discovery.ipynb` or `/workspace/src/nomad/notebooks/eda.ipynb` to start exploring the ML workflows.

