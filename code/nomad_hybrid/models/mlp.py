import torch
import torch.nn as nn

class NomadMLP(nn.Module):
    def __init__(self, tabular_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)