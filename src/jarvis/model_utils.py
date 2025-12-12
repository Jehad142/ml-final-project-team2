# model_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


# ----------------------------
# Formula embedder
# ----------------------------
class FormulaEmbedder(nn.Module):
    """
    Embeds a per-element count vector into a fixed-length representation
    via a learned per-element embedding and weighted sum.
    Input: element_counts (tensor of shape (..., num_elements))
    Output: tensor of shape (..., embed_dim)
    """
    def __init__(self, num_elements: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_elements, embed_dim)

    def forward(self, element_counts: torch.Tensor) -> torch.Tensor:
        # element_counts: (num_elements,) or (batch, num_elements)
        if element_counts.dim() == 1:
            idx = torch.arange(element_counts.shape[-1], device=element_counts.device)
            weights = element_counts.float().unsqueeze(-1)  # (num_elements, 1)
            vecs = self.emb(idx)                            # (num_elements, embed_dim)
            return (vecs * weights).sum(dim=0)              # (embed_dim,)
        elif element_counts.dim() == 2:
            B, E = element_counts.shape
            idx = torch.arange(E, device=element_counts.device)
            vecs = self.emb(idx)                            # (E, embed_dim)
            weights = element_counts.float().unsqueeze(-1)  # (B, E, 1)
            vecs = vecs.unsqueeze(0).expand(B, E, -1)       # (B, E, embed_dim)
            return (vecs * weights).sum(dim=1)              # (B, embed_dim)
        else:
            raise ValueError(f"element_counts must be 1D or 2D, got {element_counts.shape}")


# ----------------------------
# Simple graph encoder
# ----------------------------
class SimpleGraphEncoder(nn.Module):
    """
    Lightweight message-passing encoder:
    - Node init: embedding(element_id) -> atom_embed_dim
    - Two message-passing rounds with edge features (edge_feat_dim)
    - Final graph readout: mean over nodes -> (atom_embed_dim,)
    """
    def __init__(self, num_elements: int, atom_embed_dim: int, edge_feat_dim: int = 4):
        super().__init__()
        self.atom_emb = nn.Embedding(num_elements, atom_embed_dim)
        self.msg1 = nn.Linear(atom_embed_dim + edge_feat_dim, atom_embed_dim)
        self.msg2 = nn.Linear(atom_embed_dim + edge_feat_dim, atom_embed_dim)
        self.node_up1 = nn.Linear(atom_embed_dim, atom_embed_dim)
        self.node_up2 = nn.Linear(atom_embed_dim, atom_embed_dim)
        self.edge_feat_dim = edge_feat_dim

    def _normalize_edge_index(self, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_index
        if edge_index.ndim == 1:
            E = edge_index.numel() // 2
            edge_index = edge_index.view(2, E)
        elif edge_index.ndim == 2:
            if edge_index.shape[0] == 2:
                pass
            elif edge_index.shape[1] == 2:
                edge_index = edge_index.t()
            else:
                raise ValueError(f"Unexpected edge_index shape: {tuple(edge_index.shape)}")
        else:
            raise ValueError(f"Unexpected edge_index ndim: {edge_index.ndim}")
        return edge_index

    def _normalize_edge_attr(self, edge_attr: torch.Tensor, E: int) -> torch.Tensor:
        if edge_attr.numel() == 0:
            return edge_attr
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.view(E, -1)
        elif edge_attr.ndim == 2 and edge_attr.shape[0] != E and edge_attr.shape[1] == E:
            edge_attr = edge_attr.t()
        if edge_attr.ndim != 2 or edge_attr.shape[0] != E:
            raise ValueError(f"edge_attr shape mismatch: {tuple(edge_attr.shape)} vs E={E}")
        return edge_attr

    def message_pass(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x
        edge_index = self._normalize_edge_index(edge_index)
        src, dst = edge_index
        E = edge_index.shape[1]
        edge_attr = self._normalize_edge_attr(edge_attr, E)

        # Round 1
        x_src = x[src]                                 # (E, atom_embed_dim)
        m = torch.cat([x_src, edge_attr], dim=-1)      # (E, atom_embed_dim + edge_feat_dim)
        m = F.relu(self.msg1(m))                       # (E, atom_embed_dim)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        x = F.relu(self.node_up1(x + agg))

        # Round 2
        x_src = x[src]
        m = torch.cat([x_src, edge_attr], dim=-1)
        m = F.relu(self.msg2(m))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        x = F.relu(self.node_up2(x + agg))
        return x

    def forward(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        # graph: {"x": (N,), "edge_index": (2,E), "edge_attr": (E, edge_feat_dim)}
        x = self.atom_emb(graph["x"])                           # (N, atom_embed_dim)
        x = self.message_pass(x, graph["edge_index"], graph["edge_attr"])
        if x.shape[0] == 0:
            return torch.zeros(self.atom_emb.embedding_dim, device=x.device)
        return x.mean(dim=0)                                    # (atom_embed_dim,)


# ----------------------------
# Categorical embeddings
# ----------------------------
class CategoricalEmbeddings(nn.Module):
    """
    ModuleDict of embeddings for categorical columns.
    X_cat_dict should have shapes (batch, 1) or (batch,) per category.
    Output: (batch, sum(embed_dim))
    """
    def __init__(self, cat_cfg: Dict[str, Dict]):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(params["vocab_size"], params["embed_dim"])
            for name, params in cat_cfg.items()
        })
        self.out_dim = sum(params["embed_dim"] for params in cat_cfg.values())

    def forward(self, X_cat_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs: List[torch.Tensor] = []
        batch_size = None
        for name, emb in self.embeddings.items():
            x = X_cat_dict[name]
            # Normalize shape to (batch, 1)
            if x.dim() == 1:
                x = x.view(x.size(0), 1)
            elif x.dim() == 0:
                x = x.view(1, 1)
            # Embedding expects indices: (batch, 1) -> (batch, 1, embed_dim)
            e = emb(x)  # (batch, 1, embed_dim)
            e = e.squeeze(1)  # (batch, embed_dim)
            embs.append(e)
            if batch_size is None:
                batch_size = e.size(0)
        # Concatenate along feature dimension
        return torch.cat(embs, dim=-1)  # (batch, sum(embed_dim))


# ----------------------------
# Multimodal model
# ----------------------------
class CandidateNetMultimodal(nn.Module):
    """
    Multimodal MLP over concatenated features:
    - Numeric features
    - Categorical embeddings
    - Formula embedding (per-element counts)
    - Graph embedding (mean-pooled node features from message passing)
    Uses LayerNorm to support batch_size=1 safely.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config

        # Components
        self.num_proj = nn.Identity()
        self.cat_emb = CategoricalEmbeddings(config["categorical"])
        self.formula_emb = FormulaEmbedder(
            config["formula"]["num_elements"],
            config["formula"]["embed_dim"]
        )
        self.graph_enc = SimpleGraphEncoder(
            config["formula"]["num_elements"],
            config["graph"]["atom_embed_dim"],
            config["graph"]["edge_feat_dim"]
        )

        # Input dimension
        in_dim = (
            len(config["num_cols"]) +
            self.cat_emb.out_dim +
            config["formula"]["embed_dim"] +
            config["graph"]["atom_embed_dim"]
        )

        # MLP head with LayerNorm
        self.fc = nn.Sequential(
            nn.Linear(in_dim, config["mlp"]["hidden1"]),
            nn.LayerNorm(config["mlp"]["hidden1"]),
            nn.ReLU(),
            nn.Dropout(p=config["mlp"]["dropout"]),
            nn.Linear(config["mlp"]["hidden1"], config["mlp"]["hidden2"]),
            nn.LayerNorm(config["mlp"]["hidden2"]),
            nn.ReLU(),
            nn.Linear(config["mlp"]["hidden2"], 1)
        )

    def forward(self,
                X_num: torch.Tensor,
                X_cat_dict: Dict[str, torch.Tensor],
                formula_counts: torch.Tensor,
                graphs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        # Numeric
        x_num = self.num_proj(X_num.float()).view(X_num.size(0), -1)

        # Categorical
        x_cat = self.cat_emb(X_cat_dict).view(X_num.size(0), -1)

        # Formula embeddings → (batch, embed_dim)
        if formula_counts.dim() == 2:
            x_formula = self.formula_emb(formula_counts)            # (batch, embed_dim)
        else:
            x_formula = self.formula_emb(formula_counts).unsqueeze(0)

        # Graph embeddings: loop over list of dicts → (batch, embed_dim)
        graph_embs: List[torch.Tensor] = []
        for g in graphs:
            ge = self.graph_enc(g)                                  # (embed_dim,)
            graph_embs.append(ge.unsqueeze(0))                      # (1, embed_dim)
        x_graph = torch.cat(graph_embs, dim=0)                      # (batch, embed_dim)

        # Concatenate all modalities
        x = torch.cat([x_num, x_cat, x_formula, x_graph], dim=-1)   # (batch, total_dim)
        logits = self.fc(x).squeeze(-1)                             # (batch,)
        return logits

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        logits = self.forward(*args, **kwargs)
        return torch.sigmoid(logits)
