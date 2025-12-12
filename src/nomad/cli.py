import argparse
import tomli

from nomad_hybrid.train import run_training
from nomad_hybrid.predict import run_inference

def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)

def main():
    parser = argparse.ArgumentParser(description="Train or run inference with NOMAD hybrid model")
    parser.add_argument("--config", type=str, help="Path to TOML config file")

    # Training arguments
    parser.add_argument("--train_csv", type=str, help="Path to train.csv")
    parser.add_argument("--xyz_dir", type=str, help="Path to geometry directory")
    parser.add_argument("--model", type=str, choices=["mlp", "hybrid"], default=None, help="Model type")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, help="Path to save model weights after training")
    parser.add_argument("--load_path", type=str, help="Path to load model weights before training")
    parser.add_argument("--scaler_path", type=str, help="Path to save/load scaler object")

    # Inference arguments
    parser.add_argument("--predict", action="store_true", help="Run inference instead of training")
    parser.add_argument("--test_csv", type=str, help="Path to test.csv for inference")
    parser.add_argument("--output", type=str, help="Path to save predictions (CSV)")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)

        # Paths
        args.train_csv = args.train_csv or config.get("paths", {}).get("train_csv")
        args.xyz_dir = args.xyz_dir or config.get("paths", {}).get("xyz_dir")
        args.save_path = args.save_path or config.get("paths", {}).get("save_path")
        args.load_path = args.load_path or config.get("paths", {}).get("load_path")
        args.scaler_path = args.scaler_path or config.get("paths", {}).get("scaler_path")

        # Model
        args.model = args.model or config.get("model", {}).get("type")
        args.epochs = args.epochs if args.epochs is not None else config.get("model", {}).get("epochs")

        # Inference
        args.test_csv = args.test_csv or config.get("inference", {}).get("test_csv")
        args.output = args.output or config.get("inference", {}).get("output")
        args.predict = args.predict or config.get("inference", {}).get("predict", False)
    # Run training or inference
    print("Running in inference mode" if args.predict else "Running in training mode")

    if args.predict:
        run_inference(
            test_csv=args.test_csv,
            xyz_dir=args.xyz_dir,
            model_type=args.model,
            load_path=args.load_path,
            output_path=args.output,
            scaler_path=args.scaler_path
        )
    else:
        run_training(
            train_csv=args.train_csv,
            xyz_dir=args.xyz_dir,
            model_type=args.model,
            epochs=args.epochs,
            save_path=args.save_path,
            load_path=args.load_path,
            scaler_path=args.scaler_path
        )

if __name__ == "__main__":
    main()
