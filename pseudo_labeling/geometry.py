"""Geometry helpers for object-detection boxes."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


Box = Tuple[float, float, float, float]


def xyxy_to_coco(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def coco_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def clip_xyxy(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def normalize_xyxy(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = clip_xyxy(box, width, height)
    return [
        x1 / max(width, 1),
        y1 / max(height, 1),
        x2 / max(width, 1),
        y2 / max(height, 1),
    ]


def denormalize_xyxy(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return clip_xyxy([x1 * width, y1 * height, x2 * width, y2 * height], width, height)
