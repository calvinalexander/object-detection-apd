"""Turn the SDK's 442 annotated images into ROI crops we can train on.

The annotations are boxes, our classifiers take anatomical crops, so pose is run
to place head and hand ROIs and each pose person is matched to an ANNOTATED
person box by IoU. The label then comes from the annotation: a positive box whose
centre falls in that person means present, a `no_*` box means absent, neither
means the annotator did not assess that attribute for that person and the crop is
skipped -- inventing a label there is how the COCO set ended up 86% violations.

Unmatched pose detections are dropped rather than guessed at: a person the
annotator did not box carries no truth.
"""
import argparse, collections, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd.pose import build_rois
from apd.schema import GEOMETRY, assert_contract

assert_contract()
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ATT = ["head_cap", "mask", "gloves"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="artifacts/sdk/sdk/customers/apd_v3/dataset")
ap.add_argument("--out", default="data/crops_sdk")
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--min-iou", type=float, default=0.5)
a = ap.parse_args()

def iou(x, y):
    ix = max(0, min(x[2],y[2]) - max(x[0],y[0])); iy = max(0, min(x[3],y[3]) - max(x[1],y[1]))
    inter = ix*iy
    u = (x[2]-x[0])*(x[3]-x[1]) + (y[2]-y[0])*(y[3]-y[1]) - inter
    return inter/max(1e-6, u)
def centre_in(b,p):
    cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]
def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

pose = YOLO(".work/yolo11x-pose.pt")
root = Path(a.data)
out = Path(a.out)
for at in ATT:
    for cls in ("present","absent"):
        (out/at/cls).mkdir(parents=True, exist_ok=True)

imgs = sorted(p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
stats = collections.Counter()
for n, p in enumerate(imgs, 1):
    im = cv2.imread(str(p))
    if im is None: continue
    H, W = im.shape[:2]
    gtb = []
    lp = root/"labels/val"/(p.stem+".txt")
    if lp.exists():
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) < 5: continue
            c = int(f[0]); cx,cy,bw,bh = (float(v) for v in f[1:5])
            gtb.append(([(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H], NAMES[c]))
    gt_persons = [b for b,nm in gtb if nm == "person"]
    if not gt_persons: stats["no annotated person"] += 1; continue

    rr = pose.predict(im, imgsz=a.imgsz, conf=a.conf, verbose=False, device=0)[0]
    if rr.keypoints is None or not len(rr.boxes): stats["pose found nobody"] += 1; continue
    people = build_rois(rr.boxes.xyxy.cpu().numpy(), rr.keypoints.data.cpu().numpy())

    for i, pp in enumerate(people):
        best = max(gt_persons, key=lambda g: iou(pp.person_box, g))
        if iou(pp.person_box, best) < a.min_iou:
            stats["pose person unmatched"] += 1; continue
        for at in ATT:
            hp = any(nm==at and centre_in(bb, best) for bb,nm in gtb)
            hn = any(nm==f"no_{at}" and centre_in(bb, best) for bb,nm in gtb)
            if hp == hn:
                stats[f"{at}: not assessed"] += 1; continue
            cls = "present" if hp else "absent"
            stem = f"sdk_{p.stem}_{i}"
            if at == "gloves":
                for j, h in enumerate(pp.hands[:2]):
                    c_ = crop_np(im, h, GEOMETRY.hand_crop_px)
                    if c_ is None: continue
                    cv2.imwrite(str(out/at/cls/f"{stem}_h{j}.jpg"), c_); stats[f"{at}/{cls}"] += 1
            else:
                if pp.head is None or pp.head_collision: stats[f"{at}: no head roi"] += 1; continue
                c_ = crop_np(im, pp.head, GEOMETRY.head_crop_px)
                if c_ is None: continue
                cv2.imwrite(str(out/at/cls/f"{stem}.jpg"), c_); stats[f"{at}/{cls}"] += 1
    if n % 100 == 0: print(f"  {n}/{len(imgs)}", flush=True)

print(f"\nwrote {out}")
for at in ATT:
    pr = stats[f"{at}/present"]; ab = stats[f"{at}/absent"]
    print(f"  {at:9} present {pr:5d}  absent {ab:5d}   (balance {pr/max(1,pr+ab):.0%} present)")
print("\nskips:")
for k, v in sorted(stats.items()):
    if "/" not in k: print(f"  {k:28} {v}")
