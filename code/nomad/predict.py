import os
import torch
import joblib
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

from nomad_hybrid.models import HybridModel, NomadMLP
from nomad_hybrid.data import HybridNomadDataset

def run_inference(test_csv, xyz_dir, model_type, load_path, output_path, scaler_path):
    # load test data
    df = pd.read_csv(test_csv)
    if 'id' not in df.columns:
        raise ValueError("Test CSV must contain an 'id' column.")

    # prepare tabular features
    X = df.drop(columns=['id']).values
    if scaler_path==None:
        scaler = StandardScaler()
        scaler.fit(X)  # in production, we ought to load scaler from training instead
    else: 
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X) 

    # build dataset and loader
    test_ds = HybridNomadDataset(df, xyz_dir, scaler, inference=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = HybridModel if model_type == "hybrid" else NomadMLP
    model = model_cls(tabular_dim=X.shape[1], hidden_dim=64).to(device)

    if not load_path or not os.path.exists(load_path):
        raise ValueError(f"Model checkpoint not found at: {load_path}")

    print(f"Loading model weights from: {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    # run inference
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🔍 Predicting"):
            tabular, graph = batch
            tabular = tabular.to(device)
            graph = graph.to(device)
            out = model(tabular, graph) if model_type == "hybrid" else model(tabular)
            preds.append(out.cpu())

    # format and save predictions
    preds = torch.cat(preds, dim=0).numpy()
    df_out = pd.DataFrame(preds, columns=["formation_energy_ev_natom", "bandgap_energy_ev"])
    df_out.insert(0, "id", df["id"])
    df_out.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
