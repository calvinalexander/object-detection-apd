"""Mixed manual/pseudo dataset loading for student training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, Dataset

from pseudo_labeling.config import StudentTrainingConfig
from pseudo_labeling.data import CocoDetectionDataset, detection_collate


class RatioMixedDetectionDataset(Dataset):
    """Concatenate manual and pseudo datasets while preserving source metadata."""

    def __init__(self, manual: CocoDetectionDataset, pseudo: CocoDetectionDataset, pseudo_loss_weight: float) -> None:
        self.manual = manual
        self.pseudo = pseudo
        self.pseudo_loss_weight = float(pseudo_loss_weight)

    def __len__(self) -> int:
        return len(self.manual) + len(self.pseudo)

    def __getitem__(self, index: int):
        if index < len(self.manual):
            image, target = self.manual[index]
            target["source"] = "manual"
            target["loss_weight"] = torch.tensor(1.0, dtype=torch.float32)
            return image, target
        image, target = self.pseudo[index - len(self.manual)]
        target["source"] = "pseudo"
        target["loss_weight"] = torch.tensor(self.pseudo_loss_weight, dtype=torch.float32)
        return image, target


class FixedRatioBatchSampler(BatchSampler):
    """Yield batches with a fixed manual:pseudo sample ratio.

    This avoids pseudo-label volume overwhelming the scarce but more reliable
    hand-labeled DL data. For a 1:1 ratio and batch_size=16, each batch contains
    8 manual and 8 pseudo samples whenever both pools have enough examples.
    """

    def __init__(
        self,
        manual_len: int,
        pseudo_len: int,
        batch_size: int,
        ratio: Tuple[int, int] = (1, 1),
        drop_last: bool = True,
    ) -> None:
        if batch_size < sum(ratio):
            raise ValueError("batch_size must be at least the sum of the ratio parts")
        self.manual_len = manual_len
        self.pseudo_len = pseudo_len
        self.batch_size = batch_size
        self.ratio = ratio
        self.drop_last = drop_last

        ratio_total = ratio[0] + ratio[1]
        self.manual_per_batch = max(1, round(batch_size * ratio[0] / ratio_total))
        self.pseudo_per_batch = batch_size - self.manual_per_batch

    def __iter__(self) -> Iterator[List[int]]:
        manual_indices = list(range(self.manual_len))
        pseudo_indices = list(range(self.manual_len, self.manual_len + self.pseudo_len))
        random.shuffle(manual_indices)
        random.shuffle(pseudo_indices)

        manual_cursor = 0
        pseudo_cursor = 0
        while manual_cursor < len(manual_indices) and pseudo_cursor < len(pseudo_indices):
            batch = (
                manual_indices[manual_cursor : manual_cursor + self.manual_per_batch]
                + pseudo_indices[pseudo_cursor : pseudo_cursor + self.pseudo_per_batch]
            )
            if len(batch) < self.batch_size and self.drop_last:
                break
            random.shuffle(batch)
            yield batch
            manual_cursor += self.manual_per_batch
            pseudo_cursor += self.pseudo_per_batch

    def __len__(self) -> int:
        return min(
            self.manual_len // self.manual_per_batch,
            self.pseudo_len // max(1, self.pseudo_per_batch),
        )


def resize_tensor_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    image = image.unsqueeze(0)
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False).squeeze(0)


def mosaic_four(samples: Sequence[Tuple[torch.Tensor, dict]], output_size: int) -> Tuple[torch.Tensor, dict]:
    """Simple 2x2 mosaic that rescales boxes into quadrants."""

    quadrant_size = output_size // 2
    canvas = torch.zeros(3, output_size, output_size, dtype=samples[0][0].dtype)
    all_boxes: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_ignore: List[torch.Tensor] = []
    offsets = [(0, 0), (quadrant_size, 0), (0, quadrant_size), (quadrant_size, quadrant_size)]

    for (image, target), (offset_x, offset_y) in zip(samples, offsets):
        _, height, width = image.shape
        scaled = resize_tensor_image(image, (quadrant_size, quadrant_size))
        canvas[:, offset_y : offset_y + quadrant_size, offset_x : offset_x + quadrant_size] = scaled

        scale_x = quadrant_size / max(width, 1)
        scale_y = quadrant_size / max(height, 1)
        boxes = target["boxes"].clone()
        if boxes.numel():
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + offset_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + offset_y
            all_boxes.append(boxes)
            all_labels.append(target["labels"])

        ignore = target.get("ignore_regions", torch.empty(0, 4)).clone()
        if ignore.numel():
            ignore[:, [0, 2]] = ignore[:, [0, 2]] * scale_x + offset_x
            ignore[:, [1, 3]] = ignore[:, [1, 3]] * scale_y + offset_y
            all_ignore.append(ignore)

    first_target = dict(samples[0][1])
    first_target["boxes"] = torch.cat(all_boxes, dim=0) if all_boxes else torch.empty(0, 4)
    first_target["labels"] = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    first_target["ignore_regions"] = torch.cat(all_ignore, dim=0) if all_ignore else torch.empty(0, 4)
    first_target["size"] = torch.tensor([output_size, output_size], dtype=torch.long)
    first_target["source"] = "mosaic"
    return canvas, first_target


def cutmix_pair(sample_a: Tuple[torch.Tensor, dict], sample_b: Tuple[torch.Tensor, dict]) -> Tuple[torch.Tensor, dict]:
    """Detection CutMix that pastes a central patch and keeps boxes by center."""

    image_a, target_a = sample_a
    image_b, target_b = sample_b
    _, height, width = image_a.shape
    image_b = resize_tensor_image(image_b, (height, width))

    cut_w = random.randint(width // 4, width // 2)
    cut_h = random.randint(height // 4, height // 2)
    x1 = random.randint(0, max(0, width - cut_w))
    y1 = random.randint(0, max(0, height - cut_h))
    x2, y2 = x1 + cut_w, y1 + cut_h

    mixed = image_a.clone()
    mixed[:, y1:y2, x1:x2] = image_b[:, y1:y2, x1:x2]

    patch = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
    target = dict(target_a)
    boxes = [target_a["boxes"]]
    labels = [target_a["labels"]]
    if target_b["boxes"].numel():
        centers_x = (target_b["boxes"][:, 0] + target_b["boxes"][:, 2]) / 2
        centers_y = (target_b["boxes"][:, 1] + target_b["boxes"][:, 3]) / 2
        keep = (centers_x >= x1) & (centers_x <= x2) & (centers_y >= y1) & (centers_y <= y2)
        boxes.append(target_b["boxes"][keep])
        labels.append(target_b["labels"][keep])
    target["boxes"] = torch.cat(boxes, dim=0)
    target["labels"] = torch.cat(labels, dim=0)
    return mixed, target


def color_jitter_tensor(image: torch.Tensor) -> torch.Tensor:
    """Lightweight tensor color jitter for already-loaded detection samples."""

    brightness = random.uniform(0.75, 1.25)
    contrast = random.uniform(0.75, 1.25)
    saturation = random.uniform(0.8, 1.2)

    image = (image * brightness).clamp(0.0, 1.0)
    mean = image.mean(dim=(1, 2), keepdim=True)
    image = ((image - mean) * contrast + mean).clamp(0.0, 1.0)
    gray = image.mean(dim=0, keepdim=True)
    image = ((image - gray) * saturation + gray).clamp(0.0, 1.0)
    return image


class StrongAugmentedDataset(Dataset):
    """Apply Mosaic, CutMix, and color jitter around a base mixed dataset."""

    def __init__(self, base: Dataset, cfg: StudentTrainingConfig) -> None:
        self.base = base
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        if random.random() < self.cfg.mosaic_probability and len(self.base) >= 4:
            indices = [index] + random.sample(range(len(self.base)), 3)
            sample = mosaic_four([self.base[i] for i in indices], self.cfg.image_size)
        else:
            sample = self.base[index]
            if random.random() < self.cfg.cutmix_probability and len(self.base) >= 2:
                other_index = random.randrange(len(self.base))
                sample = cutmix_pair(sample, self.base[other_index])

        if random.random() < self.cfg.color_jitter_probability:
            image, target = sample
            sample = (color_jitter_tensor(image), target)
        return sample


def build_mixed_student_dataset(
    gt_coco_json: Path,
    gt_image_dir: Path,
    pseudo_coco_json: Path,
    pseudo_image_dir: Path,
    cfg: StudentTrainingConfig,
) -> Tuple[Dataset, FixedRatioBatchSampler]:
    manual = CocoDetectionDataset(gt_coco_json, gt_image_dir, source="manual")
    pseudo = CocoDetectionDataset(pseudo_coco_json, pseudo_image_dir, source="pseudo")
    base = RatioMixedDetectionDataset(manual, pseudo, pseudo_loss_weight=cfg.pseudo_loss_weight)
    augmented = StrongAugmentedDataset(base, cfg)
    sampler = FixedRatioBatchSampler(
        manual_len=len(manual),
        pseudo_len=len(pseudo),
        batch_size=cfg.batch_size,
        ratio=cfg.labeled_to_pseudo_ratio,
    )
    return augmented, sampler
