#!/usr/bin/env python
"""Smoke-test filtering and COCO export without loading teacher models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudo_labeling.config import PseudoLabelConfig
from pseudo_labeling.filtering import filter_all_predictions
from pseudo_labeling.export import export_pseudo_coco
from pseudo_labeling.models import Detection


def main() -> None:
    cfg = PseudoLabelConfig(output_json=Path("outputs/smoke_pseudo_coco.json"))
    image_records = [
        {"id": 1, "file_name": "example_a.jpg", "width": 640, "height": 480},
        {"id": 2, "file_name": "example_b.jpg", "width": 640, "height": 480},
    ]
    predictions = {
        1: [
            Detection([10, 20, 120, 220], 0.95, 1, "smoke"),
            Detection([200, 180, 250, 240], 0.30, 3, "smoke"),
            Detection([205, 185, 255, 245], 0.08, 3, "smoke"),
        ],
        2: [
            Detection([30, 30, 90, 100], 0.72, 2, "smoke"),
            Detection([400, 100, 430, 145], 0.12, 4, "smoke"),
        ],
    }
    filtered, thresholds = filter_all_predictions(predictions, cfg.filtering)
    export_pseudo_coco(image_records, filtered, cfg.output_json, info={"thresholds": thresholds})
    print(f"wrote {cfg.output_json}")
    print(thresholds)


if __name__ == "__main__":
    main()
