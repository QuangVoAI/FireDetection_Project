"""
scripts/infer_demo.py
---------------------
CLI entry point for real-time inference + alert demo.

Usage
-----
    # Webcam (default)
    python scripts/infer_demo.py

    # Video file with SAHI
    python scripts/infer_demo.py --source path/to/video.mp4 --use-sahi

    # RTSP stream, save output
    python scripts/infer_demo.py --source rtsp://... --save --output runs/out.mp4

    # Batch inference with SAHI on folder 04
    python scripts/infer_demo.py --folder data/04_SAHI_Small_Objects
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire Detection – Inference Demo")
    parser.add_argument("--weights", default="models/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", default=None,
                        help="Video source: int (webcam), file path, or RTSP URL")
    parser.add_argument("--folder", default=None,
                        help="Run SAHI batch inference on a folder of images")
    parser.add_argument("--use-sahi", action="store_true",
                        help="Use SAHI sliced inference (for small/distant objects)")
    parser.add_argument("--no-display", action="store_true", help="Suppress live window")
    parser.add_argument("--save", action="store_true", help="Save annotated output video")
    parser.add_argument("--output", default="runs/inference/output.mp4",
                        help="Output video path (used with --save)")
    args = parser.parse_args()

    if args.folder:
        from src.utils.sahi_integration import predict_folder_with_sahi
        predict_folder_with_sahi(
            folder=args.folder,
            weights=args.weights,
        )
    else:
        from src.inference import run_inference
        run_inference(
            weights=args.weights,
            source=args.source,
            use_sahi=args.use_sahi,
            display=not args.no_display,
            save_output=args.save,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
