import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ase import Atoms
from ase.neighborlist import neighbor_list

def xyz_to_graph(base_dir, material_id, cutoff=5.0):
    path = os.path.join(base_dir, str(material_id), "geometry.xyz")
    symbols, positions = [], []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith("atom"):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                symbol = parts[4]
                positions.append([x, y, z])
                symbols.append(symbol)

    atoms = Atoms(symbols=symbols, positions=positions)
    Z = atoms.get_atomic_numbers()
    i, j, _ = neighbor_list('ijS', atoms, cutoff=cutoff)

    edge_index = torch.tensor([i, j], dtype=torch.long)
    x = torch.tensor(Z, dtype=torch.float32).unsqueeze(1)
    pos = torch.tensor(positions, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, pos=pos)

class HybridNomadDataset(Dataset):
    def __init__(self, df, base_dir, scaler=None):
        self.df = df.copy()
        self.base_dir = base_dir
        self.scaler = scaler

        self.tabular = df.drop(columns=['id', 'formation_energy_ev_natom', 'bandgap_energy_ev']).values
        if scaler:
            self.tabular = scaler.transform(self.tabular)

        self.targets = df[['formation_energy_ev_natom', 'bandgap_energy_ev']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        material_id = int(row['id'])
        tabular_feat = torch.tensor(self.tabular[idx], dtype=torch.float32)
        graph = xyz_to_graph(self.base_dir, material_id)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return tabular_feat, graph, y