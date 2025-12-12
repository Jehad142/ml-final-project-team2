import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class HybridModel(nn.Module):
    def __init__(self, tabular_dim, hidden_dim=64):
        super().__init__()
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, tabular, graph):
        tab_out = self.tabular_net(tabular)
        x, edge_index, batch = graph.x.float(), graph.edge_index, graph.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        graph_out = global_mean_pool(x, batch)
        combined = torch.cat([tab_out, graph_out], dim=1)
        return self.fusion(combined)
