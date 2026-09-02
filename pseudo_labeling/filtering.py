"""Dynamic class thresholding and ignore-region mining."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import CATEGORY_ID_TO_NAME, FilteringConfig
from .geometry import box_area, iou_xyxy
from .models import Detection


@dataclass(frozen=True)
class FilteredDetections:
    accepted: List[Detection]
    ignore_regions: List[Detection]


def percentile(values: Sequence[float], pct: float) -> float:
    """Small dependency-free percentile implementation."""

    if not values:
        return 1.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def compute_class_thresholds(
    predictions_by_image: Dict[int, Sequence[Detection]],
    cfg: FilteringConfig,
) -> Dict[int, float]:
    """Compute per-class score thresholds from global teacher distributions."""

    scores_by_class: Dict[int, List[float]] = defaultdict(list)
    for detections in predictions_by_image.values():
        for det in detections:
            scores_by_class[det.category_id].append(float(det.score))

    thresholds: Dict[int, float] = {}
    for category_id, scores in scores_by_class.items():
        class_name = CATEGORY_ID_TO_NAME[category_id]
        keep_top_percent = cfg.keep_top_percent_by_class[class_name]
        threshold_percentile = max(0.0, min(100.0, 100.0 - keep_top_percent))
        dynamic_threshold = percentile(scores, threshold_percentile)
        floor = cfg.min_score_floor_by_class.get(class_name, 0.0)
        thresholds[category_id] = max(floor, dynamic_threshold)
    return thresholds


def overlaps_accepted(det: Detection, accepted: Sequence[Detection], max_iou: float) -> bool:
    return any(iou_xyxy(det.box_xyxy, good.box_xyxy) > max_iou for good in accepted)


def filter_predictions_for_image(
    detections: Sequence[Detection],
    thresholds: Dict[int, float],
    cfg: FilteringConfig,
) -> FilteredDetections:
    """Split WBF consensus into final pseudo labels and ambiguity masks."""

    accepted: List[Detection] = []
    candidates_for_ignore: List[Detection] = []

    for det in sorted(detections, key=lambda item: item.score, reverse=True):
        if box_area(det.box_xyxy) < cfg.min_box_area_px:
            continue
        threshold = thresholds.get(det.category_id, 1.0)
        if det.score >= threshold:
            accepted.append(det)
        elif det.score >= cfg.ignore_score_floor:
            candidates_for_ignore.append(det)

    ignore_regions = [
        det
        for det in candidates_for_ignore
        if not overlaps_accepted(det, accepted, cfg.ignore_max_iou_with_accepted)
    ]
    return FilteredDetections(accepted=accepted, ignore_regions=ignore_regions)


def filter_all_predictions(
    predictions_by_image: Dict[int, Sequence[Detection]],
    cfg: FilteringConfig,
) -> Tuple[Dict[int, FilteredDetections], Dict[int, float]]:
    """Compute thresholds globally, then filter every image consistently."""

    thresholds = compute_class_thresholds(predictions_by_image, cfg)
    filtered = {
        image_id: filter_predictions_for_image(detections, thresholds, cfg)
        for image_id, detections in predictions_by_image.items()
    }
    return filtered, thresholds
