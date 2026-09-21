"""Train YOLO26 on the person-crop attribute dataset and report per-class results.

The dataset is one image per person, so a "detection" here is an attribute on
that person, not a person in a scene. Two consequences shape the settings:

  - objects are small relative to the crop (a mask is a fraction of a head) and
    already centred, so mosaic hurts more than it helps late in training
  - the split is grouped by site and by video, so val is genuinely held out and
    early stopping on it is meaningful

Per-class mAP is what matters, not the mean: the mean is carried by head_cap,
which has three times the boxes of mask.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/det7/dataset.yaml")
ap.add_argument("--models", nargs="+", default=["yolo26n.pt", "yolo26s.pt"])
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--project", default="runs/det7")
ap.add_argument("--aug", choices=["crop", "frame"], default="crop",
                help="crop: the person fills the image, so mosaic and big shifts push attributes out of "
                     "frame. frame: a normal scene, where mosaic helps and objects are small.")
ap.add_argument("--device", default="0")
a = ap.parse_args()

from ultralytics import YOLO
NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
results = {}

for w in a.models:
    tag = Path(w).stem
    print(f"\n=== {tag} ===", flush=True)
    t0 = time.perf_counter()
    m = YOLO(w)
    m.train(data=str(ROOT/a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            patience=a.patience, project=str(ROOT/a.project), name=tag, exist_ok=True,
            device=a.device, workers=8, seed=20260916, val=True, plots=True,
            **(dict(mosaic=0.4, close_mosaic=20, scale=0.35, translate=0.08, degrees=4.0)
               if a.aug == "crop" else
               # full frames: mosaic is worth having, and CCTV is always upright
               dict(mosaic=0.8, close_mosaic=20, scale=0.5, translate=0.1, degrees=0.0)),
            fliplr=0.5, flipud=0.0, hsv_v=0.4, hsv_s=0.5, erasing=0.1)
    mins = (time.perf_counter()-t0)/60
    mt = m.val(data=str(ROOT/a.data), imgsz=a.imgsz, device=a.device, verbose=False)
    per = {}
    for i, c in enumerate(mt.ap_class_index):
        per[NAMES[int(c)]] = {"P": float(mt.box.p[i]), "R": float(mt.box.r[i]),
                              "mAP50": float(mt.box.ap50[i]), "mAP50-95": float(mt.box.ap[i])}
    results[tag] = {"minutes": round(mins, 1), "mAP50": float(mt.box.map50),
                    "mAP50-95": float(mt.box.map), "per_class": per,
                    "params_M": round(sum(p.numel() for p in m.model.parameters())/1e6, 2)}
    print(f"{tag}: mAP50 {mt.box.map50:.3f}  mAP50-95 {mt.box.map:.3f}  ({mins:.1f} min)", flush=True)

out = ROOT/a.project/"summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))

print(f"\n{'model':10} {'params':>7} {'mAP50':>7} {'mAP50-95':>9}   per-class mAP50")
for tag, r in results.items():
    pc = "  ".join(f"{k}={v['mAP50']:.2f}" for k, v in r["per_class"].items())
    print(f"{tag:10} {r['params_M']:6.1f}M {r['mAP50']:7.3f} {r['mAP50-95']:9.3f}   {pc}")
print(f"\nwrote {out}")
