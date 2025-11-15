## Usage

### Setup

```bash
cd ~/src/ml-final-project-team2/code
# or set the module path explicitly
export PYTHONPATH=~/src/ml-final-project-team2/code
```

---

### Train Models

#### Hybrid Model (Tabular + Graph)

```bash
python -m nomad_hybrid.cli \
  --csv /shared/data/nomad2018/train.csv \
  --xyz_dir /shared/data/nomad2018/train \
  --model hybrid \
  --epochs 30 \
  --save_path /shared/data/checkpoints/hybrid.pt
```

#### MLP Model (Tabular Only)

```bash
python -m nomad_hybrid.cli \
  --csv /shared/data/nomad2018/train.csv \
  --xyz_dir /shared/data/nomad2018/train \
  --model mlp \
  --epochs 1 \
  --save_path /shared/data/checkpoints/mlp.pt
```

---

### 🔍 Run Inference

#### Using Hybrid Model

```bash
python -m nomad_hybrid.cli \
  --predict \
  --test_csv /shared/data/nomad2018/test.csv \
  --xyz_dir /shared/data/nomad2018/test \
  --model hybrid \
  --load_path /shared/data/checkpoints/hybrid.pt \
  --output predictions.csv
```

#### Using MLP Model

```bash
python -m nomad_hybrid.cli \
  --predict \
  --test_csv /shared/data/nomad2018/test.csv \
  --xyz_dir /shared/data/nomad2018/test \
  --model mlp \
  --load_path /shared/data/checkpoints/mlp.pt \
  --output predictions_mlp.csv
```

## Project Manifest

```bash
$ tree -I '__pycache__' -I '*.csv'
.
└── nomad_hybrid
    ├── README.md
    ├── __init__.py
    ├── cli.py
    ├── data
    │   ├── __init__.py
    │   └── dataset.py
    ├── models
    │   ├── __init__.py
    │   ├── hybrid.py
    │   └── mlp.py
    ├── predict.py
    └── train.py

3 directories, 10 files
```

### Top-Level Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, usage instructions, and documentation |
| `__init__.py` | Marks `nomad_hybrid` as a Python package |
| `cli.py` | Command-line interface for training and inference; parses arguments and dispatches to core functions |
| `train.py` | Training pipeline: loads data, initializes model, runs training loop, saves best checkpoint |
| `predict.py` | Inference pipeline: loads test data and model weights, generates predictions, saves output |

---

### `data/` Module

| File | Purpose |
|------|---------|
| `__init__.py` | Initializes the `data` submodule |
| `dataset.py` | Defines `HybridNomadDataset` for tabular + graph data loading; supports training and inference modes |

---

### `models/` Module

| File | Purpose |
|------|---------|
| `__init__.py` | Initializes the `models` submodule |
| `hybrid.py` | Defines `HybridModel`: combines tabular MLP and graph neural network for joint prediction |
| `mlp.py` | Defines `NomadMLP`: a lightweight model for tabular-only prediction |
