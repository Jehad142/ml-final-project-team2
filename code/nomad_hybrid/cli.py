import argparse
from nomad_hybrid.train import run_training
from nomad_hybrid.predict import run_inference

def main():
    parser = argparse.ArgumentParser(description="Train hybrid NOMAD model")
    parser.add_argument("--csv", type=str, help="Path to train.csv")
    parser.add_argument("--xyz_dir", type=str, required=True, help="Path to geometry directory")
    parser.add_argument("--model", type=str, choices=["mlp", "hybrid"], default="hybrid", help="Model type")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, help="Path to save model weights after training")
    parser.add_argument("--load_path", type=str, help="Path to load model weights before training")

    parser.add_argument("--predict", action="store_true", help="Run inference instead of training")
    parser.add_argument("--test_csv", type=str, help="Path to test.csv for inference")
    parser.add_argument("--output", type=str, help="Path to save predictions (CSV)")

    args = parser.parse_args()
    #run_training(args.csv, args.xyz_dir, args.model, args.epochs, args.save_path, args.load_path)
    if args.predict:
        run_inference(args.test_csv, args.xyz_dir, args.model, args.load_path, args.output)
    else:
        run_training(args.csv, args.xyz_dir, args.model, args.epochs, args.save_path, args.load_path)

if __name__ == "__main__":
    main()
