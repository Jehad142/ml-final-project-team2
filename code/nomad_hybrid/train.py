import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from nomad_hybrid.models import HybridModel, NomadMLP
from nomad_hybrid.data import HybridNomadDataset

#def run_training(csv_path, xyz_dir, model_type="hybrid", epochs=30):
def run_training(csv_path, xyz_dir, model_type="hybrid", epochs=30, save_path=None, load_path=None):
    print(f"Loading data from: {csv_path}")
    print(f"Geometry directory: {xyz_dir}")
    print(f"Model type: {model_type}")

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
    if device.type == "cpu":
        print("️Warning: Training is running on CPU. This may be significantly slower than using a GPU.")

    print(f"Training for {epochs} epochs on {device.type.upper()}")

    model_cls = HybridModel if model_type == "hybrid" else NomadMLP
    model = model_cls(tabular_dim=X.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()
    if load_path and os.path.exists(load_path):
        print(f"Loading model weights from {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))

    def train_epoch():
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc="🟢 Training", leave=False)
        for tabular, graph, y in loop:
            tabular, graph, y = tabular.to(device), graph.to(device), y.to(device)
            preds = model(tabular, graph) if model_type == "hybrid" else model(tabular)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return total_loss / len(train_loader)

    def validate_epoch():
        model.eval()
        total_loss = 0
        loop = tqdm(val_loader, desc="🔵 Validation", leave=False)
        with torch.no_grad():
            for tabular, graph, y in loop:
                tabular, graph, y = tabular.to(device), graph.to(device), y.to(device)
                preds = model(tabular, graph) if model_type == "hybrid" else model(tabular)
                loss = criterion(preds, y)
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
        return total_loss / len(val_loader)

    best_val_loss = float("inf")
    best_model_state = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch()
        val_loss = validate_epoch()
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end=" ")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            print("(New best model)")
        else:
            print("")

    if save_path and best_model_state:
        print(f"Saving best model (val loss {best_val_loss:.4f}) to: {save_path}")
        torch.save(best_model_state, save_path)