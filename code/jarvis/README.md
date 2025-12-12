```
project_root/
│
├── config_multimodal.toml        # TOML config
├── notebooks/
│   └── training.ipynb            # thin orchestration notebook
│
├── data_utils.py                 # dataset loading, candidate filtering, vocab building
├── model_utils.py                # embedders, graph encoder, multimodal model
├── train_utils.py                # training loop, checkpointing
├── eval_utils.py                 # evaluation, metrics, confusion matrix
├── plot_utils.py                 # plotting + saving figures with captions
├── logger_utils.py               # logger setup/flush
└── jarvis_utils.py               # dataset fetch wrapper (already referenced)
```