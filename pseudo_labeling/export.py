"""COCO export for pseudo labels and ignore regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .config import category_records
from .filtering import FilteredDetections
from .geometry import box_area, xyxy_to_coco
from .models import Detection


def _annotation_record(
    ann_id: int,
    image_id: int,
    det: Detection,
    ignore: bool,
) -> dict:
    bbox = xyxy_to_coco(det.box_xyxy)
    record = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": int(det.category_id),
        "bbox": bbox,
        "area": float(bbox[2] * bbox[3]),
        "iscrowd": 1 if ignore else 0,
        "ignore": 1 if ignore else 0,
        "score": float(det.score),
        "pseudo": True,
    }
    if ignore:
        record["loss_mask"] = True
    return record


def export_pseudo_coco(
    image_records: Sequence[Mapping],
    filtered_by_image: Dict[int, FilteredDetections],
    output_json: Path,
    info: Mapping | None = None,
) -> None:
    """Write pseudo labels to a COCO-style JSON file.

    Ambiguous regions are represented as annotations with ignore=1 and iscrowd=1.
    Downstream loaders can use these boxes to mask loss for anchors/queries that
    overlap uncertain teacher evidence.
    """

    output_json.parent.mkdir(parents=True, exist_ok=True)
    annotations: List[dict] = []
    ann_id = 1
    for image in image_records:
        image_id = int(image["id"])
        filtered = filtered_by_image.get(image_id)
        if filtered is None:
            continue

        for det in filtered.accepted:
            annotations.append(_annotation_record(ann_id, image_id, det, ignore=False))
            ann_id += 1
        for det in filtered.ignore_regions:
            annotations.append(_annotation_record(ann_id, image_id, det, ignore=True))
            ann_id += 1

    coco = {
        "info": dict(info or {"description": "SPPG offline pseudo labels"}),
        "licenses": [],
        "categories": category_records(),
        "images": [dict(image) for image in image_records],
        "annotations": annotations,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
