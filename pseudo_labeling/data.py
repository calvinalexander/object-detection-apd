"""COCO parsing and dataset utilities for labeled and unlabeled images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import CATEGORY_NAME_TO_ID, CLASS_NAMES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CocoImageRecord:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoAnnotationRecord:
    id: int
    image_id: int
    category_id: int
    bbox: Tuple[float, float, float, float]
    area: float
    iscrowd: int = 0
    ignore: int = 0
    score: Optional[float] = None


def load_coco_json(coco_json: Path) -> dict:
    """Load a COCO JSON file with a short validation pass."""

    with coco_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for required_key in ("images", "annotations", "categories"):
        if required_key not in data:
            raise ValueError(f"{coco_json} is missing required COCO key: {required_key}")
    return data


def build_coco_api(coco_json: Path):
    """Create a pycocotools COCO object for metrics or external tooling."""

    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise ImportError("Install pycocotools to use COCO API helpers.") from exc
    return COCO(str(coco_json))


def project_category_mapping(coco: dict, class_names: Sequence[str] = CLASS_NAMES) -> Dict[int, int]:
    """Map source COCO category IDs onto the canonical four-class IDs.

    COCO Annotator may assign arbitrary category IDs. This function keeps only
    the target classes and remaps them into 1..4 so teacher/student exports are
    stable across annotation exports.
    """

    wanted = set(class_names)
    source_name_by_id = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
    mapping: Dict[int, int] = {}
    for source_id, name in source_name_by_id.items():
        if name in wanted:
            mapping[source_id] = CATEGORY_NAME_TO_ID[name]

    missing = sorted(wanted - set(source_name_by_id.values()))
    if missing:
        raise ValueError(f"Ground-truth COCO file is missing target categories: {missing}")
    return mapping


def parse_coco_records(coco_json: Path, class_names: Sequence[str] = CLASS_NAMES) -> Tuple[List[CocoImageRecord], Dict[int, List[CocoAnnotationRecord]]]:
    """Parse image and annotation records for the requested classes only."""

    coco = load_coco_json(coco_json)
    category_id_map = project_category_mapping(coco, class_names)

    images = [
        CocoImageRecord(
            id=int(img["id"]),
            file_name=str(img["file_name"]),
            width=int(img.get("width", 0)),
            height=int(img.get("height", 0)),
        )
        for img in coco["images"]
    ]
    annotations_by_image: Dict[int, List[CocoAnnotationRecord]] = {img.id: [] for img in images}

    for ann in coco["annotations"]:
        source_category_id = int(ann["category_id"])
        if source_category_id not in category_id_map:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w <= 0 or h <= 0:
            continue
        image_id = int(ann["image_id"])
        annotations_by_image.setdefault(image_id, []).append(
            CocoAnnotationRecord(
                id=int(ann["id"]),
                image_id=image_id,
                category_id=category_id_map[source_category_id],
                bbox=(x, y, w, h),
                area=float(ann.get("area", w * h)),
                iscrowd=int(ann.get("iscrowd", 0)),
                ignore=int(ann.get("ignore", 0)),
                score=ann.get("score"),
            )
        )
    return images, annotations_by_image


def resolve_image_path(image_dir: Path, file_name: str) -> Path:
    """Resolve COCO file_name values that may be relative or nested."""

    candidate = Path(file_name)
    if candidate.is_absolute():
        return candidate
    return image_dir / candidate


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to float tensor in C,H,W range [0,1]."""

    image = image.convert("RGB")
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).contiguous()
    return data.float().div_(255.0)


class CocoDetectionDataset(Dataset):
    """PyTorch dataset for manually labeled COCO data."""

    def __init__(
        self,
        coco_json: Path,
        image_dir: Path,
        class_names: Sequence[str] = CLASS_NAMES,
        transform: Optional[Callable] = None,
        source: str = "manual",
    ) -> None:
        self.image_dir = image_dir
        self.images, self.annotations_by_image = parse_coco_records(coco_json, class_names)
        self.transform = transform
        self.source = source

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        record = self.images[index]
        image_path = resolve_image_path(self.image_dir, record.file_name)
        image = Image.open(image_path).convert("RGB")

        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        ignore_regions: List[List[float]] = []
        scores: List[float] = []

        for ann in self.annotations_by_image.get(record.id, []):
            x, y, w, h = ann.bbox
            xyxy = [x, y, x + w, y + h]
            if ann.ignore:
                ignore_regions.append(xyxy)
                continue
            boxes_xyxy.append(xyxy)
            labels.append(ann.category_id)
            scores.append(float(ann.score) if ann.score is not None else 1.0)

        target = {
            "image_id": torch.tensor(record.id, dtype=torch.long),
            "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "ignore_regions": torch.tensor(ignore_regions, dtype=torch.float32).reshape(-1, 4),
            "size": torch.tensor([image.height, image.width], dtype=torch.long),
            "path": str(image_path),
            "source": self.source,
        }

        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = pil_to_tensor(image)
        return image, target


class UnlabeledImageDataset(Dataset):
    """Dataset over a directory tree of raw images for teacher generation."""

    def __init__(self, image_dir: Path, transform: Optional[Callable] = None) -> None:
        self.image_dir = image_dir
        self.image_paths = sorted(
            path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No images found under unlabeled directory: {image_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Image.Image, dict]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        meta = {
            "image_id": index + 1,
            "file_name": str(path.relative_to(self.image_dir)),
            "path": str(path),
            "width": image.width,
            "height": image.height,
        }
        if self.transform:
            image, meta = self.transform(image, meta)
        return image, meta


def detection_collate(batch: Iterable[Tuple[torch.Tensor, dict]]) -> Tuple[List[torch.Tensor], List[dict]]:
    """Collate detection samples without padding boxes."""

    images, targets = zip(*batch)
    return list(images), list(targets)


def unlabeled_collate(batch):
    """Keep PIL images and metadata as lists for teacher inference."""

    images, metas = zip(*batch)
    return list(images), list(metas)


def build_labeled_loader(
    coco_json: Path,
    image_dir: Path,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create DL: the manually labeled ground-truth detection loader."""

    dataset = CocoDetectionDataset(coco_json, image_dir, source="manual")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
    )


def build_unlabeled_loader(
    image_dir: Path,
    batch_size: int = 1,
    num_workers: int = 4,
) -> DataLoader:
    """Create DU: the raw unlabeled-image loader for teacher generation."""

    dataset = UnlabeledImageDataset(image_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=unlabeled_collate,
        pin_memory=False,
    )
