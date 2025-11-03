import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader

from nomad_hybrid.models import HybridModel, NomadMLP
from nomad_hybrid.data import HybridNomadDataset

def run_training(csv_path, xyz_dir, model_type="hybrid", epochs=30):
    # Load and preprocess data
    df = pd.read_csv(csv_path)
    scaler = StandardScaler()
    X = df.drop(columns=['id', 'formation_energy_ev_natom', 'bandgap_energy_ev']).values
    scaler.fit(X)

    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    train_ds = HybridNomadDataset(df_train, xyz_dir, scaler)
    val_ds = HybridNomadDataset(df_val, xyz_dir, scaler)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = HybridModel if model_type == "hybrid" else NomadMLP
    model = model_cls(tabular_dim=X.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()

    def train_epoch():
        model.train()
        total_loss = 0
        for tabular, graph, y in train_loader:
            tabular, graph, y = tabular.to(device), graph.to(device), y.to(device)
            preds = model(tabular, graph) if model_type == "hybrid" else model(tabular)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate_epoch():
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for tabular, graph, y in val_loader:
                tabular, graph, y = tabular.to(device), graph.to(device), y.to(device)
                preds = model(tabular, graph) if model_type == "hybrid" else model(tabular)
                loss = criterion(preds, y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch()
        val_loss = validate_epoch()
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

