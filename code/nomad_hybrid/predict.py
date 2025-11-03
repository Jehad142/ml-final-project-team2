import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from nomad_hybrid.models import HybridModel, NomadMLP
from nomad_hybrid.data import HybridNomadDataset
from sklearn.preprocessing import StandardScaler

def run_inference(test_csv, xyz_dir, model_type, load_path, output_path):
    df = pd.read_csv(test_csv)
    scaler = StandardScaler()
    X = df.drop(columns=['id']).values
    scaler.fit(X)  # or load from training if saved

    test_ds = HybridNomadDataset(df, xyz_dir, scaler)
    test_loader = DataLoader(test_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = HybridModel if model_type == "hybrid" else NomadMLP
    model = model_cls(tabular_dim=X.shape[1], hidden_dim=64).to(device)

    if not load_path or not os.path.exists(load_path):
        raise ValueError("Must provide valid --load_path for inference")

    print(f"Loading model weights from: {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for tabular, graph, _ in test_loader:
            tabular, graph = tabular.to(device), graph.to(device)
            out = model(tabular, graph) if model_type == "hybrid" else model(tabular)
            preds.append(out.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    df_out = pd.DataFrame(preds, columns=["formation_energy_ev_natom", "bandgap_energy_ev"])
    df_out.insert(0, "id", df["id"])
    df_out.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
