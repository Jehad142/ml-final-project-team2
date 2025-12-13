# MSSE Machine Learning Final Project – Team 2 Submission

This project is delivered as a Dockerized environment.  
If Docker is installed, setup and execution are simple: clone the repository and run `make` from the root directory.

---

## Clone the Repository

```bash
git clone https://github.com/Jehad142/ml-final-project-team2.git
cd ml-final-project-team2
```

---

## Build and Run with Docker

From the project root, run:

```bash
make
```

This will:
- Build the Docker image (`chem277b-final:2025.12`) using the configuration in `docker/Dockerfile`.
- Launch a container (`chem277b-session`) with the project mounted at `/workspace`.
- Start JupyterLab on port `8888` with authentication disabled for local development.

Once running, open your browser at:

```
http://127.0.0.1:8888/lab
```

---

## Quick Test

Navigate to the following notebooks inside JupyterLab:

- `/workspace/src/jarvis/notebooks/discovery.ipynb`
- `/workspace/src/nomad/notebooks/eda.ipynb`

Run the cells to verify the environment and explore the ML workflows.

---

## Clean Up

To remove containers, images, and unused resources:

```bash
make clean
```

---

## Notes

- Docker handles all dependencies automatically via `docker/requirements.txt`.  
- No local Python environment setup is required.  
- If Docker is not available, follow the manual setup instructions in `README.md`.

