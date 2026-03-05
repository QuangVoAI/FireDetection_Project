"""
scripts/evaluate.py
-------------------
CLI entry point for model evaluation.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --weights models/weights/best.pt --split test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire Detection – Model Evaluation")
    parser.add_argument("--weights", default="models/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--dataset", default="config/dataset.yaml", help="Dataset YAML")
    parser.add_argument("--config", default="config/model_config.yaml", help="Model config YAML")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--output-dir", default="runs/evaluate", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    metrics = evaluate(
        weights=args.weights,
        dataset_yaml=args.dataset,
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )

    print("\n── Evaluation Results ─────────────────────────")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
    print("───────────────────────────────────────────────")


if __name__ == "__main__":
    main()
