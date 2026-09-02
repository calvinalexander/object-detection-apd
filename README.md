# SPPG Offline Pseudo-Labeling Framework

Production-oriented scaffold for one-off multiclass object-detection dataset expansion.

Target classes:

- `person`
- `gloves`
- `mask`
- `head_cap`

Ground truth:

`/data/coco-annotator-prod/coco-annotator/datasets/sppg/.exports/coco-1777022473.3698506.json`

Images:

`/data/coco-annotator-prod/coco-annotator/datasets/sppg/`

## Pipeline

1. Parse COCO and canonicalize class IDs to `1..4`.
2. Run a high-precision teacher ensemble over an unlabeled image directory.
3. Apply multi-scale and horizontal-flip TTA.
4. Fuse teacher/TTA predictions with Weighted Boxes Fusion.
5. Compute global per-class percentile thresholds.
6. Export accepted boxes plus low-confidence ambiguous boxes as COCO ignore regions.
7. Train a deployable student with fixed-ratio manual/pseudo mixed batches and strong augmentation.

## Generate Pseudo Labels

Install dependencies:

```bash
pip install -r requirements.txt
```

If you do not already have a trained teacher checkpoint, first bootstrap one
from the existing labeled COCO export:

```bash
python scripts/prepare_yolo_seed_dataset.py \
  --output-dir datasets/yolo_seed

python scripts/train_yolo_seed.py \
  --data datasets/yolo_seed/data.yaml \
  --model yolov8x.pt \
  --image-size 1536 \
  --batch 2 \
  --name sppg_seed_teacher
```

The initial teacher checkpoint will be written to:

```text
runs/detect/sppg_seed_teacher/weights/best.pt
```

Wire the RF-DETR adapter in `pseudo_labeling/models.py` to your RF-DETR runtime, or remove it from `PseudoLabelConfig.teachers` if you only want to run YOLO teachers.

Run generation:

```bash
python scripts/generate_pseudo_labels.py \
  --unlabeled-dir /path/to/unlabeled/images \
  --yolo-checkpoint runs/detect/sppg_seed_teacher/weights/best.pt \
  --output-json outputs/DU_pseudo_coco.json
```

You can pass `--yolo-checkpoint` multiple times to ensemble several YOLO
teachers. If omitted, the script uses the default teachers configured in
`pseudo_labeling/config.py`.

Rerun only thresholding/export after editing filtering settings:

```bash
python scripts/generate_pseudo_labels.py \
  --unlabeled-dir /path/to/unlabeled/images \
  --raw-jsonl outputs/teacher_raw_predictions.jsonl \
  --output-json outputs/DU_pseudo_coco.json \
  --reuse-raw
```

## Student Training Scaffold

`student/training_config.py` provides a framework-neutral training loop contract. Replace `StudentDetectorAdapter` with the project YOLO nano/small implementation or connect it to your detector framework's native trainer.

The mixed loader preserves:

- Manual:pseudo batch ratio, default `1:1`.
- `loss_weight` per sample for pseudo-label down-weighting.
- `ignore_regions` from ambiguous teacher boxes for anchor/query loss masking.
- Mosaic, CutMix, and color jitter augmentation.

```bash
python scripts/train_student.py \
  --pseudo-json outputs/DU_pseudo_coco.json \
  --pseudo-image-dir /path/to/unlabeled/images
```
