"""End-to-end teacher generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from torch.utils.data import DataLoader

from .config import PseudoLabelConfig
from .data import UnlabeledImageDataset
from .export import export_pseudo_coco
from .filtering import filter_all_predictions
from .models import Detection, TeacherModelAdapter, build_teacher_ensemble
from .tta import predict_consensus_sources_with_tta
from .wbf import weighted_fusion


def detection_to_dict(det: Detection) -> dict:
    return {
        "box_xyxy": [float(v) for v in det.box_xyxy],
        "score": float(det.score),
        "category_id": int(det.category_id),
        "model_name": det.model_name,
    }


def detection_from_dict(raw: dict) -> Detection:
    return Detection(
        box_xyxy=[float(v) for v in raw["box_xyxy"]],
        score=float(raw["score"]),
        category_id=int(raw["category_id"]),
        model_name=str(raw.get("model_name", "")),
    )


def generate_raw_teacher_predictions(
    unlabeled_dir: Path,
    cfg: PseudoLabelConfig,
    teachers: Sequence[TeacherModelAdapter] | None = None,
) -> Tuple[List[dict], Dict[int, List[Detection]]]:
    """Run teacher ensemble + TTA + WBF and return raw consensus detections."""

    dataset = UnlabeledImageDataset(unlabeled_dir)
    teachers = list(teachers or build_teacher_ensemble(cfg.teachers, cfg.class_names))
    image_records: List[dict] = []
    predictions_by_image: Dict[int, List[Detection]] = {}

    cfg.raw_prediction_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with cfg.raw_prediction_jsonl.open("w", encoding="utf-8") as raw_file:
        for image, meta in dataset:
            image_id = int(meta["image_id"])
            image_records.append(
                {
                    "id": image_id,
                    "file_name": meta["file_name"],
                    "width": int(meta["width"]),
                    "height": int(meta["height"]),
                }
            )

            consensus_sources, source_weights = predict_consensus_sources_with_tta(teachers, image, cfg.tta)
            fused = weighted_fusion(
                detections_by_model=consensus_sources,
                model_weights=source_weights,
                width=image.width,
                height=image.height,
                cfg=cfg.wbf,
            )
            predictions_by_image[image_id] = fused
            raw_file.write(
                json.dumps(
                    {
                        "image": image_records[-1],
                        "detections": [detection_to_dict(det) for det in fused],
                    }
                )
                + "\n"
            )
    return image_records, predictions_by_image


def load_raw_teacher_predictions(raw_jsonl: Path) -> Tuple[List[dict], Dict[int, List[Detection]]]:
    """Load a previous raw prediction JSONL to rerun thresholding cheaply."""

    image_records: List[dict] = []
    predictions_by_image: Dict[int, List[Detection]] = {}
    with raw_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            image = row["image"]
            image_id = int(image["id"])
            image_records.append(image)
            predictions_by_image[image_id] = [detection_from_dict(det) for det in row["detections"]]
    return image_records, predictions_by_image


def run_generation(
    unlabeled_dir: Path,
    cfg: PseudoLabelConfig,
    reuse_raw_predictions: bool = False,
) -> Dict[int, float]:
    """Run the full offline pseudo-labeling pipeline and export COCO JSON."""

    if reuse_raw_predictions and cfg.raw_prediction_jsonl.exists():
        image_records, predictions_by_image = load_raw_teacher_predictions(cfg.raw_prediction_jsonl)
    else:
        image_records, predictions_by_image = generate_raw_teacher_predictions(unlabeled_dir, cfg)

    filtered_by_image, thresholds = filter_all_predictions(predictions_by_image, cfg.filtering)
    export_pseudo_coco(
        image_records=image_records,
        filtered_by_image=filtered_by_image,
        output_json=cfg.output_json,
        info={
            "description": "SPPG offline pseudo labels generated with teacher ensemble, TTA, WBF, and dynamic class thresholds",
            "thresholds_by_category_id": {str(k): v for k, v in thresholds.items()},
        },
    )
    return thresholds
