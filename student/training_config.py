"""Student model training configuration and adapter skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from pseudo_labeling.config import CLASS_NAMES, GT_COCO_JSON, IMAGE_DIR, StudentTrainingConfig
from pseudo_labeling.data import detection_collate
from .dataset import build_mixed_student_dataset


class StudentDetectorAdapter(torch.nn.Module):
    """Framework adapter for a lightweight deployable student detector.

    Wire this class to YOLO nano/small, RT-DETR-R18, or another deployable model.
    The forward method should consume a list/tensor of images and targets and
    return a loss dictionary during training.
    """

    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        raise RuntimeError(
            "StudentDetectorAdapter is a training-loop contract. Attach your "
            "YOLO nano/small implementation here or replace this class with the "
            "student framework's native trainer."
        )

    def forward(self, images, targets=None):
        raise NotImplementedError


def build_student_dataloader(
    pseudo_coco_json: Path,
    pseudo_image_dir: Path,
    cfg: StudentTrainingConfig,
) -> DataLoader:
    """Create a 1:1 manual:pseudo mixed dataloader with strong augmentations."""

    dataset, batch_sampler = build_mixed_student_dataset(
        gt_coco_json=GT_COCO_JSON,
        gt_image_dir=IMAGE_DIR,
        pseudo_coco_json=pseudo_coco_json,
        pseudo_image_dir=pseudo_image_dir,
        cfg=cfg,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
    )


def train_student(
    pseudo_coco_json: Path,
    pseudo_image_dir: Path,
    cfg: StudentTrainingConfig = StudentTrainingConfig(),
) -> None:
    """Reference training loop showing where loss masks and source weights apply."""

    dataloader = build_student_dataloader(pseudo_coco_json, pseudo_image_dir, cfg)
    model = StudentDetectorAdapter(cfg.student_model_name, num_classes=len(CLASS_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(cfg.epochs):
        for images, targets in dataloader:
            # A production YOLO adapter should use:
            # - target["loss_weight"] to down-weight pseudo-label loss
            # - target["ignore_regions"] to mask anchors/queries overlapping
            #   ambiguous teacher boxes
            # - strong augmentations already applied by the dataset wrapper
            loss_dict: Dict[str, torch.Tensor] = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1}/{cfg.epochs} loss={float(loss.detach().cpu()):.4f}")
