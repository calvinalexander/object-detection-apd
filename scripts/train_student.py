#!/usr/bin/env python
"""Run the reference mixed-batch student training loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudo_labeling.config import StudentTrainingConfig
from student.training_config import train_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-json", type=Path, required=True)
    parser.add_argument("--pseudo-image-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StudentTrainingConfig(batch_size=args.batch_size, epochs=args.epochs)
    train_student(args.pseudo_json, args.pseudo_image_dir, cfg)


if __name__ == "__main__":
    main()
