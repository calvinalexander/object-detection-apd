"""Teacher model adapter interfaces.

The framework deliberately does not hard-code a single detector runtime. RF-DETR,
Ultralytics YOLO, MMDetection, Detectron2, or an internal inference service can
all be plugged in by implementing TeacherModelAdapter.predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from PIL import Image

from .config import CLASS_NAMES, TeacherModelConfig


@dataclass(frozen=True)
class Detection:
    """One detection in absolute xyxy pixel coordinates."""

    box_xyxy: List[float]
    score: float
    category_id: int
    model_name: str = ""


class TeacherModelAdapter:
    """Base class for high-precision teacher detectors."""

    def __init__(self, cfg: TeacherModelConfig, class_names: Sequence[str] = CLASS_NAMES) -> None:
        self.cfg = cfg
        self.class_names = tuple(class_names)

    def predict(self, image: Image.Image) -> List[Detection]:
        """Run inference on an RGB PIL image.

        Subclasses return boxes in the coordinate space of the provided image.
        """

        raise NotImplementedError


class RFDETRAdapter(TeacherModelAdapter):
    """Pseudo-code wrapper for RF-DETR style APIs.

    Replace the import/model construction with the exact RF-DETR package used in
    your environment. The adapter output contract is the only thing the rest of
    the pipeline depends on.
    """

    def __init__(self, cfg: TeacherModelConfig, class_names: Sequence[str] = CLASS_NAMES) -> None:
        super().__init__(cfg, class_names)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        # Example placeholder:
        # from rfdetr import RFDETR
        # self.model = RFDETR.from_pretrained(self.cfg.checkpoint).to(self.cfg.device).eval()
        raise RuntimeError(
            "RFDETRAdapter is a production-ready interface but needs your RF-DETR "
            "runtime import/checkpoint wiring. Edit pseudo_labeling/models.py::_load_model."
        )

    def predict(self, image: Image.Image) -> List[Detection]:
        # Example placeholder:
        # raw = self.model.predict(image, image_size=self.cfg.image_size)
        # return [
        #     Detection(box_xyxy=det.xyxy, score=det.score, category_id=det.class_id + 1, model_name=self.cfg.name)
        #     for det in raw
        #     if det.class_name in self.class_names
        # ]
        raise NotImplementedError


class UltralyticsYOLOAdapter(TeacherModelAdapter):
    """Adapter for Ultralytics YOLO models.

    The package is imported lazily so the rest of the codebase can be used
    without installing Ultralytics on systems that only post-process predictions.
    """

    def __init__(self, cfg: TeacherModelConfig, class_names: Sequence[str] = CLASS_NAMES) -> None:
        super().__init__(cfg, class_names)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install ultralytics to use UltralyticsYOLOAdapter.") from exc

        self.model = YOLO(cfg.checkpoint)
        self.name_to_project_id = {name: idx + 1 for idx, name in enumerate(self.class_names)}

    def predict(self, image: Image.Image) -> List[Detection]:
        results = self.model.predict(
            source=image,
            imgsz=self.cfg.image_size,
            conf=0.001,
            iou=0.9,
            device=self.cfg.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for xyxy, conf, cls_idx in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                class_name = names[int(cls_idx)]
                if class_name not in self.name_to_project_id:
                    continue
                detections.append(
                    Detection(
                        box_xyxy=[float(v) for v in xyxy],
                        score=float(conf),
                        category_id=self.name_to_project_id[class_name],
                        model_name=self.cfg.name,
                    )
                )
        return detections


class NoOpTeacherAdapter(TeacherModelAdapter):
    """Tiny adapter for dry-runs and tests."""

    def predict(self, image: Image.Image) -> List[Detection]:
        return []


def build_teacher_model(cfg: TeacherModelConfig, class_names: Sequence[str] = CLASS_NAMES) -> TeacherModelAdapter:
    """Construct a teacher adapter from a config descriptor."""

    model_type = cfg.model_type.lower()
    if model_type in {"ultralytics_yolo", "yolo", "yolov8", "yolov9", "yolo11"}:
        return UltralyticsYOLOAdapter(cfg, class_names)
    if model_type in {"rfdetr", "rf-detr", "rf_detr"}:
        return RFDETRAdapter(cfg, class_names)
    if model_type in {"noop", "dry_run"}:
        return NoOpTeacherAdapter(cfg, class_names)
    raise ValueError(f"Unsupported teacher model_type={cfg.model_type!r}")


def build_teacher_ensemble(configs: Iterable[TeacherModelConfig], class_names: Sequence[str] = CLASS_NAMES) -> List[TeacherModelAdapter]:
    return [build_teacher_model(cfg, class_names) for cfg in configs]
