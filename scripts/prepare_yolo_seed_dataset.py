#!/usr/bin/env python
"""Convert the existing SPPG COCO annotations into an Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudo_labeling.config import CLASS_NAMES, GT_COCO_JSON, IMAGE_DIR
from pseudo_labeling.data import parse_coco_records, resolve_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", type=Path, default=GT_COCO_JSON)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/yolo_seed"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them.")
    return parser.parse_args()


def link_or_copy_image(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def yolo_line(category_id: int, bbox_xywh: Sequence[float], image_width: int, image_height: int) -> str:
    """Convert canonical one-based COCO category IDs to zero-based YOLO labels."""

    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = (x + w / 2.0) / max(image_width, 1)
    cy = (y + h / 2.0) / max(image_height, 1)
    nw = w / max(image_width, 1)
    nh = h / max(image_height, 1)
    values = [category_id - 1, cx, cy, nw, nh]
    return f"{values[0]} {values[1]:.8f} {values[2]:.8f} {values[3]:.8f} {values[4]:.8f}"


def write_data_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    content = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names}\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    images, annotations_by_image = parse_coco_records(args.coco_json)

    shuffled = list(images)
    random.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * args.val_fraction)) if len(shuffled) > 1 else 0
    val_ids = {record.id for record in shuffled[:val_count]}

    for record in images:
        split = "val" if record.id in val_ids else "train"
        src = resolve_image_path(args.image_dir, record.file_name)
        if not src.exists():
            raise FileNotFoundError(f"Image referenced by COCO does not exist: {src}")

        image_dst = args.output_dir / "images" / split / src.name
        label_dst = args.output_dir / "labels" / split / f"{src.stem}.txt"
        link_or_copy_image(src, image_dst, args.copy_images)

        lines: List[str] = []
        for ann in annotations_by_image.get(record.id, []):
            lines.append(yolo_line(ann.category_id, ann.bbox, record.width, record.height))
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_data_yaml(args.output_dir)
    print(f"Wrote YOLO seed dataset to {args.output_dir}")
    print(f"Train images: {len(images) - len(val_ids)}")
    print(f"Val images: {len(val_ids)}")
    print(f"Data YAML: {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
