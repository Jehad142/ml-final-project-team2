import re
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from jarvis.core.atoms import Atoms

# ----------------------------
# Candidate filtering
# ----------------------------
def add_candidate_column(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    bandgap_col   = config["filters"]["bandgap_column"]
    sem_min       = config["filters"]["semiconductor_min"]
    sem_max       = config["filters"]["semiconductor_max"]
    trans_min     = config["filters"]["transparent_min"]
    toxic_elements = config["filters"]["toxic_elements"]

    df[bandgap_col] = pd.to_numeric(df[bandgap_col], errors="coerce")
    in_semiconductor_range = df[bandgap_col].between(sem_min, sem_max)
    is_transparent = df[bandgap_col] > trans_min

    if "ehull" in df.columns:
        df["ehull"] = pd.to_numeric(df["ehull"], errors="coerce")
        is_stable = df["ehull"] < config["filters"]["ehull"]
    else:
        is_stable = True

    if "formula" in df.columns:
        tokens = df["formula"].fillna("").astype(str).str.findall(r"[A-Z][a-z]?")
        has_toxic = tokens.apply(lambda t: any(el in t for el in toxic_elements))
    else:
        has_toxic = False

    df["target"] = (
        in_semiconductor_range &
        is_transparent &
        is_stable &
        (has_toxic == False)
    ).astype(int)

    return df

# ----------------------------
# Formula utilities
# ----------------------------
def build_element_vocab(df, max_elements=89):
    elems = set()
    if "formula" in df.columns:
        tokens = df["formula"].fillna("").astype(str).str.findall(r"[A-Z][a-z]?")
        for t in tokens:
            elems.update(t)
    elems = sorted(list(elems))
    vocab = {}
    start_idx = 0
    if len(elems) + 1 > max_elements:
        vocab["UNK"] = 0
        start_idx = 1
    for i, e in enumerate(elems[: max_elements - start_idx]):
        vocab[e] = i + start_idx
    return vocab

def formula_to_counts(formula, element_vocab, num_elements):
    counts = np.zeros(num_elements, dtype=np.float32)
    if not isinstance(formula, str):
        return counts
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula)
    for sym, num in tokens:
        idx = element_vocab.get(sym, element_vocab.get("UNK", None))
        if idx is None:
            continue
        val = float(num) if num not in ("", None) else 1.0
        counts[idx] += val
    return counts

# ----------------------------
# Graph building
# ----------------------------
def atoms_to_graph(atoms_obj, radius=5.0, max_neighbors=None,
                   element_vocab=None, num_elements=89, device="cpu"):
    cart_coords = np.array(atoms_obj.cart_coords)
    species = atoms_obj.elements
    N = len(species)

    node_idx = np.array([element_vocab.get(sym, element_vocab.get("UNK", 0)) for sym in species], dtype=np.int64)

    edge_src, edge_dst, edge_attr = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rij = cart_coords[j] - cart_coords[i]
            dist = np.linalg.norm(rij)
            if dist <= radius:
                inv = 1.0 / dist if dist > 1e-8 else 0.0
                unit = rij / dist if dist > 1e-8 else np.zeros(3)
                feat = [dist, inv, unit[0], unit[1]]
                edge_src.append(i); edge_dst.append(j); edge_attr.append(feat)

    # Cap neighbors if requested
    if max_neighbors is not None and len(edge_src) > 0:
        capped_src, capped_dst, capped_attr = [], [], []
        edges_by_src = {}
        for s, d, a in zip(edge_src, edge_dst, edge_attr):
            edges_by_src.setdefault(s, []).append((d, a))
        for s, lst in edges_by_src.items():
            lst_sorted = sorted(lst, key=lambda x: x[1][0])[:max_neighbors]
            for d, a in lst_sorted:
                capped_src.append(s); capped_dst.append(d); capped_attr.append(a)
        edge_src, edge_dst, edge_attr = capped_src, capped_dst, capped_attr

    x = torch.tensor(node_idx, dtype=torch.long, device=device)
    if len(edge_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr_t = torch.empty((0, 4), dtype=torch.float32, device=device)
    else:
        edge_index = torch.stack([
            torch.tensor(edge_src, dtype=torch.long, device=device),
            torch.tensor(edge_dst, dtype=torch.long, device=device)
        ], dim=0)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr_t}

# ----------------------------
# Dataset + collate
# ----------------------------
class MaterialsDataset(Dataset):
    def __init__(self, frame, config, element_vocab, num_elements, device="cpu"):
        self.df = frame.reset_index(drop=True)
        self.cfg = config
        self.element_vocab = element_vocab
        self.num_elements = num_elements
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        X_num = torch.tensor(
            row[self.cfg["num_cols"]].astype(float).values,
            dtype=torch.float32,
            device=self.device
        )

        X_cat_dict = {
            name: torch.tensor(row[f"{name}_idx"], dtype=torch.long, device=self.device)
            for name in self.cfg["categorical"].keys()
        }

        counts = torch.tensor(row["formula_counts"], dtype=torch.float32, device=self.device)

        aobj = row.get("atoms_obj", None)
        if aobj is None:
            graph = {
                "x": torch.empty((0,), dtype=torch.long, device=self.device),
                "edge_index": torch.empty((2, 0), dtype=torch.long, device=self.device),
                "edge_attr": torch.empty((0, self.cfg["graph"]["edge_feat_dim"]),
                                         dtype=torch.float32, device=self.device)
            }
        else:
            graph = atoms_to_graph(
                aobj,
                radius=self.cfg["graph"]["radius"],
                max_neighbors=self.cfg["graph"]["max_neighbors"],
                element_vocab=self.element_vocab,
                num_elements=self.num_elements,
                device=self.device
            )

        y = torch.tensor(row["target"], dtype=torch.float32, device=self.device)

        return X_num, X_cat_dict, counts, graph, y

def collate_single(batch):
    X_num_list, X_cat_dicts, counts_list, graphs, y_list = zip(*batch)
    X_num = torch.stack(X_num_list)
    counts = torch.stack(counts_list)
    y = torch.stack(y_list)

    X_cat_dict = {}
    for name in X_cat_dicts[0].keys():
        X_cat_dict[name] = torch.stack([d[name] for d in X_cat_dicts]).view(len(batch), -1)

    graphs = list(graphs)
    return X_num, X_cat_dict, counts, graphs, y
