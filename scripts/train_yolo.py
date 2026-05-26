from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detection or segmentation model.")
    parser.add_argument("--data", required=True, type=Path, help="Ultralytics dataset yaml.")
    parser.add_argument("--model", required=True, help="Base weights, e.g. yolo11s.pt or yolo11s-seg.pt.")
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--name", required=True, help="Run name under outputs/runs.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", default="auto")
    parser.add_argument("--device", default=None, help="Example: 0, 0,1, cpu, mps.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    model = YOLO(args.model, task=args.task)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project="outputs/runs",
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
