#!/usr/bin/env python
"""Train the initial YOLO teacher checkpoint from the small labeled seed set."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/yolo_seed/data.yaml"))
    parser.add_argument(
        "--model",
        default="yolov8x.pt",
        help="Ultralytics model/checkpoint to fine-tune. Use yolov8x.pt or another large model for teacher quality.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="sppg_seed_teacher")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics first: pip install ultralytics") from exc

    if not args.data.exists():
        raise FileNotFoundError(
            f"YOLO data config not found: {args.data}. "
            "Run scripts/prepare_yolo_seed_dataset.py first."
        )

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=50,
        optimizer="AdamW",
        lr0=1e-4,
        weight_decay=5e-4,
        mosaic=0.8,
        mixup=0.15,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        close_mosaic=20,
        cache=False,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print("Training complete.")
    print("Use this teacher checkpoint for pseudo-label generation:")
    print(best)
    print(results)


if __name__ == "__main__":
    main()
