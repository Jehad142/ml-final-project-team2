
This repository contains code, configuration, and notebooks for training and evaluating a multimodal graph neural network (GNN) model to identify candidate transparent semiconductors. The workflow integrates tabular descriptors, compositional embeddings, and graph encodings of crystal structures.

---

## Project Structure

```
project_root/
│
├── config_multimodal.toml        # Central TOML config for dataset paths, hyperparameters, and training options
├── notebooks/
│   ├── discovery.ipynb           # Orchestration notebook for candidate discovery pipeline
│   ├── eda.ipynb                 # Exploratory data analysis (EDA) notebook
│   └── training.ipynb            # Orchestration notebook for model training
│
├── data_utils.py                 # Dataset loading, candidate filtering, vocabulary building
├── model_utils.py                # Embedders, graph encoder, multimodal model definitions
├── train_utils.py                # Training loop, checkpointing, optimizer setup
├── eval_utils.py                 # Evaluation routines, metrics, confusion matrix generation
├── plot_utils.py                 # Plotting utilities, figure saving with captions
├── logger_utils.py               # Logger setup and flush utilities
└── jarvis_utils.py               # Dataset fetch wrapper for JARVIS integration
```

---

