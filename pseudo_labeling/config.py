"""Central configuration for pseudo-label generation.

Keep paths and precision-oriented defaults here so generation runs are
reproducible. The defaults favor label quality over recall and compute speed,
which matches a one-off offline dataset expansion workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


CLASS_NAMES: Tuple[str, ...] = ("person", "gloves", "mask", "head_cap")
CATEGORY_NAME_TO_ID: Dict[str, int] = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}
CATEGORY_ID_TO_NAME: Dict[int, str] = {idx: name for name, idx in CATEGORY_NAME_TO_ID.items()}

GT_COCO_JSON = Path(
    "/data/coco-annotator-prod/coco-annotator/datasets/sppg/.exports/"
    "coco-1777022473.3698506.json"
)
IMAGE_DIR = Path("/data/coco-annotator-prod/coco-annotator/datasets/sppg/")


@dataclass(frozen=True)
class TTAConfig:
    """Test-time augmentation settings.

    Multi-scale inference plus horizontal flips increases compute but improves
    agreement quality for small PPE classes such as gloves and masks.
    """

    scales: Sequence[float] = (0.75, 1.0, 1.25, 1.5)
    horizontal_flip: bool = True


@dataclass(frozen=True)
class WBFConfig:
    """Weighted Boxes Fusion settings.

    A relatively high IoU threshold prevents distinct nearby objects from being
    merged, while skip_box_thr stays low because filtering is handled later via
    global per-class confidence percentiles.
    """

    iou_thr: float = 0.62
    skip_box_thr: float = 0.001
    conf_type: str = "avg"
    allows_overflow: bool = False


@dataclass(frozen=True)
class FilteringConfig:
    """Dynamic class thresholding and ignore-region settings.

    keep_top_percent_by_class accepts the top X percent of model consensus
    scores per class. Rare classes can safely use a larger top percentage to
    reduce false negatives while still avoiding a static threshold that is
    unfair to class-imbalanced score distributions.
    """

    keep_top_percent_by_class: Dict[str, float] = field(
        default_factory=lambda: {
            "person": 20.0,
            "gloves": 35.0,
            "mask": 35.0,
            "head_cap": 35.0,
        }
    )
    min_score_floor_by_class: Dict[str, float] = field(
        default_factory=lambda: {
            "person": 0.25,
            "gloves": 0.15,
            "mask": 0.15,
            "head_cap": 0.15,
        }
    )
    ignore_score_floor: float = 0.03
    ignore_max_iou_with_accepted: float = 0.5
    min_box_area_px: float = 16.0


@dataclass(frozen=True)
class TeacherModelConfig:
    """Descriptor for one teacher model adapter.

    model_type is resolved by pseudo_labeling.models.build_teacher_model.
    checkpoint can be a path, model name, or registry key depending on adapter.
    """

    name: str
    model_type: str
    checkpoint: str
    weight: float = 1.0
    device: str = "cuda"
    image_size: int = 1536


@dataclass(frozen=True)
class PseudoLabelConfig:
    """Top-level generation configuration."""

    gt_coco_json: Path = GT_COCO_JSON
    image_dir: Path = IMAGE_DIR
    class_names: Sequence[str] = CLASS_NAMES
    tta: TTAConfig = field(default_factory=TTAConfig)
    wbf: WBFConfig = field(default_factory=WBFConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    teachers: Sequence[TeacherModelConfig] = field(
        default_factory=lambda: (
            TeacherModelConfig(
                name="yolo_x_1536",
                model_type="ultralytics_yolo",
                checkpoint="runs/detect/train/weights/best.pt",
                weight=1.0,
                image_size=1536,
            ),
            TeacherModelConfig(
                name="yolo_x_1920",
                model_type="ultralytics_yolo",
                checkpoint="runs/detect/train2/weights/best.pt",
                weight=1.0,
                image_size=1920,
            ),
        )
    )
    output_json: Path = Path("outputs/DU_pseudo_coco.json")
    raw_prediction_jsonl: Path = Path("outputs/teacher_raw_predictions.jsonl")


@dataclass(frozen=True)
class StudentTrainingConfig:
    """Student training defaults for mixed real/pseudo batches."""

    student_model_name: str = "yolo_nano_or_small"
    batch_size: int = 16
    labeled_to_pseudo_ratio: Tuple[int, int] = (1, 1)
    epochs: int = 150
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    num_workers: int = 8
    image_size: int = 960
    mosaic_probability: float = 0.6
    cutmix_probability: float = 0.25
    color_jitter_probability: float = 0.8
    pseudo_loss_weight: float = 0.5
    ignore_regions_enabled: bool = True


def category_records() -> List[dict]:
    """Return COCO category records in the canonical project order."""

    return [{"id": idx + 1, "name": name, "supercategory": "sppg"} for idx, name in enumerate(CLASS_NAMES)]
