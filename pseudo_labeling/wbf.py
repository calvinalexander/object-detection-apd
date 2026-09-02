"""Weighted Boxes Fusion consensus for teacher/TTA predictions."""

from __future__ import annotations

from typing import List, Sequence

from .config import WBFConfig
from .geometry import denormalize_xyxy, normalize_xyxy
from .models import Detection


def weighted_fusion(
    detections_by_model: Sequence[Sequence[Detection]],
    model_weights: Sequence[float],
    width: int,
    height: int,
    cfg: WBFConfig,
) -> List[Detection]:
    """Fuse boxes from teacher models and TTA using ensemble-boxes WBF.

    ensemble-boxes expects normalized xyxy boxes and zero-based labels. The rest
    of this codebase uses absolute xyxy boxes and one-based COCO category IDs.
    """

    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as exc:
        raise ImportError(
            "Install ensemble-boxes for high-quality consensus: "
            "pip install ensemble-boxes"
        ) from exc

    boxes_list: List[List[List[float]]] = []
    scores_list: List[List[float]] = []
    labels_list: List[List[int]] = []

    for detections in detections_by_model:
        boxes_list.append([normalize_xyxy(det.box_xyxy, width, height) for det in detections])
        scores_list.append([float(det.score) for det in detections])
        labels_list.append([int(det.category_id) - 1 for det in detections])

    if not any(boxes_list):
        return []

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list=boxes_list,
        scores_list=scores_list,
        labels_list=labels_list,
        weights=list(model_weights),
        iou_thr=cfg.iou_thr,
        skip_box_thr=cfg.skip_box_thr,
        conf_type=cfg.conf_type,
        allows_overflow=cfg.allows_overflow,
    )

    fused: List[Detection] = []
    for box, score, label in zip(boxes, scores, labels):
        fused.append(
            Detection(
                box_xyxy=denormalize_xyxy(box, width, height),
                score=float(score),
                category_id=int(label) + 1,
                model_name="wbf_consensus",
            )
        )
    return fused
