"""Offline pseudo-labeling framework for SPPG object detection.

The package is intentionally modular: teacher models are adapters, consensus is
isolated behind Weighted Boxes Fusion, and filtering/export are independent of
the model framework used to produce raw detections.
"""

from .config import (
    CLASS_NAMES,
    CATEGORY_NAME_TO_ID,
    IMAGE_DIR,
    GT_COCO_JSON,
    PseudoLabelConfig,
)

__all__ = [
    "CLASS_NAMES",
    "CATEGORY_NAME_TO_ID",
    "IMAGE_DIR",
    "GT_COCO_JSON",
    "PseudoLabelConfig",
]
