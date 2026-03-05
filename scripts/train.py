"""
scripts/train.py
----------------
CLI entry point for model training.

Usage
-----
    # Full two-stage pipeline (baseline → hard-negative mining)
    python scripts/train.py

    # Baseline only
    python scripts/train.py --stage baseline

    # Hard-negative mining only (requires baseline weights)
    python scripts/train.py --stage hard_negative_mining

    # Resume interrupted training
    python scripts/train.py --resume
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_training import train, train_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire Detection – Model Training")
    parser.add_argument("--config", default="config/model_config.yaml", help="Model config YAML")
    parser.add_argument("--dataset", default="config/dataset.yaml", help="Dataset YAML")
    parser.add_argument(
        "--stage",
        choices=["baseline", "hard_negative_mining", "full"],
        default="full",
        help="Training stage to run (default: full = baseline + hard_negative_mining)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    if args.stage == "full":
        train_full_pipeline(config_path=args.config, dataset_yaml=args.dataset)
    else:
        train(
            config_path=args.config,
            dataset_yaml=args.dataset,
            stage=args.stage,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
