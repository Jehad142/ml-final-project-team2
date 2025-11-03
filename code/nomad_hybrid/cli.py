import argparse
from nomad_hybrid.train import run_training

def main():
    parser = argparse.ArgumentParser(description="Train hybrid NOMAD model")
    parser.add_argument("--csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--xyz_dir", type=str, required=True, help="Path to geometry directory")
    parser.add_argument("--model", type=str, choices=["mlp", "hybrid"], default="hybrid", help="Model type")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()

    run_training(args.csv, args.xyz_dir, args.model, args.epochs)

if __name__ == "__main__":
    main()
