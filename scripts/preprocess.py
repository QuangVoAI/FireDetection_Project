"""
scripts/preprocess.py
---------------------
CLI entry point for the preprocessing pipeline.

Usage
-----
    python scripts/preprocess.py
    python scripts/preprocess.py --data-root data --size 640
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire Detection – Data Preprocessing")
    parser.add_argument("--data-root", default="data", help="Project data root directory")
    parser.add_argument("--config", default="config/model_config.yaml", help="Model config YAML")
    parser.add_argument("--size", type=int, default=640, help="Target image size (square)")
    parser.add_argument("--hash-threshold", type=int, default=8,
                        help="Perceptual hash distance threshold for deduplication (0–64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    run_preprocessing(
        data_root=args.data_root,
        config_path=args.config,
        target_size=args.size,
        hash_threshold=args.hash_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
