# Docker Setup for ml-final-project-team2

This directory contains the Docker configuration and helper Makefile for building and running the project container.

---

## Building the Image

To build the Docker image defined in `Dockerfile`:

```bash
make build
```

This will produce an image tagged as `chem277b-final:2025.12`.

To force a rebuild without using the cache:

```bash
make rebuild
```

---

## Running the Container

To launch the container and start JupyterLab on port 8888:

```bash
make run
```

This will:
- Mount the project root into `/workspace` inside the container.
- Enable GPU support (`--gpus all`).
- Expose JupyterLab at `http://127.0.0.1:8888/lab`.
- Disable authentication for local development (`--NotebookApp.token=''`).

Stop the container with `Ctrl+C`.

---

## Additional Targets

- `foo`: Example target to launch JupyterLab with `README.md` as the initial URL.
- `moo`: Example target to export the current JupyterLab workspace to `workspace.json`.

---

## Cleaning Up

To remove the container, image, and prune unused Docker resources:

```bash
make clean
```

This will:
- Remove the running container (`chem277b-session`).
- Remove the image (`chem277b-final:2025.12`).
- Prune unused Docker resources.

