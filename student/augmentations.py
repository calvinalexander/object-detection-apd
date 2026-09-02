"""Strong detection augmentations for student training.

The implementation is intentionally framework-neutral PyTorch/PIL code so the
same mixed loader can feed a custom student, a YOLO adapter, or a distillation
experiment. Mosaic and CutMix are implemented at sample-composition level in
student.dataset because they need access to multiple images.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
from PIL import Image, ImageEnhance

try:
    from torchvision.transforms import ColorJitter
except ImportError:  # pragma: no cover - exercised only when torchvision is absent.
    ColorJitter = None

from pseudo_labeling.data import pil_to_tensor


class DetectionColorJitter:
    """Color-only augmentation that leaves boxes unchanged."""

    def __init__(self, probability: float = 0.8) -> None:
        self.probability = probability
        self.jitter = (
            ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04)
            if ColorJitter is not None
            else None
        )

    def _pil_fallback(self, image: Image.Image) -> Image.Image:
        """Approximate ColorJitter when torchvision is unavailable."""

        brightness = random.uniform(0.75, 1.25)
        contrast = random.uniform(0.75, 1.25)
        saturation = random.uniform(0.8, 1.2)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        return image

    def __call__(self, image: Image.Image, target: dict) -> Tuple[torch.Tensor, dict]:
        if random.random() < self.probability:
            image = self.jitter(image) if self.jitter is not None else self._pil_fallback(image)
        return pil_to_tensor(image), target
