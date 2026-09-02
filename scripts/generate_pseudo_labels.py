#!/usr/bin/env python
"""Generate offline pseudo labels for an unlabeled image directory."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudo_labeling.config import PseudoLabelConfig, TeacherModelConfig
from pseudo_labeling.pipeline import run_generation


def looks_like_local_path(checkpoint: str) -> bool:
    """Return True for checkpoint strings that should exist on disk."""

    path = Path(checkpoint)
    return path.is_absolute() or "/" in checkpoint or checkpoint.startswith(".")


def validate_teacher_checkpoints(teachers: Sequence[TeacherModelConfig]) -> None:
    """Fail early with a clear message for missing local teacher weights."""

    missing = []
    for teacher in teachers:
        if looks_like_local_path(teacher.checkpoint) and not Path(teacher.checkpoint).exists():
            missing.append((teacher.name, teacher.checkpoint))
    if not missing:
        return

    lines = ["Missing teacher checkpoint file(s):"]
    lines.extend(f"  - {name}: {checkpoint}" for name, checkpoint in missing)
    lines.append("")
    lines.append("Pass the real YOLO checkpoint with --yolo-checkpoint /absolute/path/to/best.pt")
    lines.append("or edit PseudoLabelConfig.teachers in pseudo_labeling/config.py.")
    raise FileNotFoundError("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlabeled-dir", type=Path, required=True, help="Directory containing raw unlabeled images.")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/DU_pseudo_coco.json"))
    parser.add_argument("--raw-jsonl", type=Path, default=Path("outputs/teacher_raw_predictions.jsonl"))
    parser.add_argument(
        "--yolo-checkpoint",
        action="append",
        help=(
            "YOLO teacher checkpoint/model name. Can be passed multiple times. "
            "When set, this overrides the teachers in pseudo_labeling/config.py."
        ),
    )
    parser.add_argument("--device", default=None, help="Teacher inference device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--image-size", type=int, default=1536, help="YOLO teacher inference image size.")
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Skip teacher inference and rerun only filtering/export from --raw-jsonl.",
    )
    return parser.parse_args()


def build_cli_teachers(args: argparse.Namespace) -> Sequence[TeacherModelConfig] | None:
    if not args.yolo_checkpoint:
        return None
    return tuple(
        TeacherModelConfig(
            name=f"yolo_teacher_{idx}",
            model_type="ultralytics_yolo",
            checkpoint=checkpoint,
            weight=1.0,
            device=args.device or "cuda",
            image_size=args.image_size,
        )
        for idx, checkpoint in enumerate(args.yolo_checkpoint, start=1)
    )


def main() -> None:
    args = parse_args()
    cli_teachers = build_cli_teachers(args)
    cfg = replace(
        PseudoLabelConfig(),
        output_json=args.output_json,
        raw_prediction_jsonl=args.raw_jsonl,
        teachers=cli_teachers or PseudoLabelConfig().teachers,
    )
    if not args.reuse_raw:
        validate_teacher_checkpoints(cfg.teachers)
    thresholds = run_generation(args.unlabeled_dir, cfg, reuse_raw_predictions=args.reuse_raw)
    print("Exported pseudo-labels to:", cfg.output_json)
    print("Dynamic thresholds by category_id:", thresholds)


if __name__ == "__main__":
    main()
