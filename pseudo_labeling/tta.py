"""Test-time augmentation for teacher inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from PIL import Image, ImageOps

from .config import TTAConfig
from .geometry import clip_xyxy
from .models import Detection, TeacherModelAdapter


@dataclass(frozen=True)
class TTASample:
    image: Image.Image
    scale: float
    flipped: bool
    original_width: int
    original_height: int


def build_tta_samples(image: Image.Image, cfg: TTAConfig) -> List[TTASample]:
    """Create scaled/flipped images for exhaustive offline inference."""

    samples: List[TTASample] = []
    original_width, original_height = image.size
    for scale in cfg.scales:
        scaled_size = (max(1, round(original_width * scale)), max(1, round(original_height * scale)))
        scaled = image.resize(scaled_size, resample=Image.Resampling.BICUBIC)
        samples.append(
            TTASample(
                image=scaled,
                scale=float(scale),
                flipped=False,
                original_width=original_width,
                original_height=original_height,
            )
        )
        if cfg.horizontal_flip:
            samples.append(
                TTASample(
                    image=ImageOps.mirror(scaled),
                    scale=float(scale),
                    flipped=True,
                    original_width=original_width,
                    original_height=original_height,
                )
            )
    return samples


def invert_tta_detection(detection: Detection, sample: TTASample) -> Detection:
    """Map a TTA-space detection back to original image coordinates."""

    x1, y1, x2, y2 = detection.box_xyxy
    tta_width = sample.image.width
    if sample.flipped:
        x1, x2 = tta_width - x2, tta_width - x1

    inv_scale = 1.0 / sample.scale
    restored = [
        x1 * inv_scale,
        y1 * inv_scale,
        x2 * inv_scale,
        y2 * inv_scale,
    ]
    return Detection(
        box_xyxy=clip_xyxy(restored, sample.original_width, sample.original_height),
        score=detection.score,
        category_id=detection.category_id,
        model_name=detection.model_name,
    )


def predict_with_tta(model: TeacherModelAdapter, image: Image.Image, cfg: TTAConfig) -> List[Detection]:
    """Run one model over all TTA samples and restore detections."""

    restored: List[Detection] = []
    for sample in build_tta_samples(image, cfg):
        for det in model.predict(sample.image):
            restored.append(invert_tta_detection(det, sample))
    return restored


def predict_ensemble_with_tta(
    models: Sequence[TeacherModelAdapter],
    image: Image.Image,
    cfg: TTAConfig,
) -> List[List[Detection]]:
    """Return one restored detection list per teacher model."""

    return [predict_with_tta(model, image, cfg) for model in models]


def predict_consensus_sources_with_tta(
    models: Sequence[TeacherModelAdapter],
    image: Image.Image,
    cfg: TTAConfig,
) -> Tuple[List[List[Detection]], List[float]]:
    """Return one WBF source per model/TTA view.

    WBF works best when each augmented view acts like an independent detector
    vote. Model weights are repeated across that model's TTA views.
    """

    sources: List[List[Detection]] = []
    weights: List[float] = []
    samples = build_tta_samples(image, cfg)
    for model in models:
        for sample in samples:
            restored = [invert_tta_detection(det, sample) for det in model.predict(sample.image)]
            sources.append(restored)
            weights.append(float(model.cfg.weight))
    return sources, weights
