"""Export pose-derived crops as a YOLO classification dataset.

Same pipeline as voter3-ours -- pose, anatomical ROI, crop -- but the crops are
written to disk in folder-per-class layout so a YOLO classifier can be trained on
them instead of a frozen-feature probe. Three independent binary problems:

    crops_cls/<attribute>/{train,val}/{present,absent}/*.jpg

Labels come from the production annotation's explicit negatives, and the split is
inherited from the camera-grouped one, so no camera appears on both sides.
Stage 1 uses yolo11n-pose @640, which measured both faster and more accurate than
the yolo11x @1920 we had been using.
"""
import collections, shutil, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd.pose import build_rois, frontality
from apd.schema import GEOMETRY

POS = {"head_cap":2, "mask":3, "gloves":1}
NEG = {"head_cap":5, "mask":6, "gloves":4}
SRC = Path("data/voter3"); OUT = Path("data/crops_cls")
pose = YOLO(".work/yolo11n-pose.pt")

def crop_of(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))
def inside(b, roi):
    cx, cy = b[0]+b[2]/2, b[1]+b[3]/2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]

if OUT.exists(): shutil.rmtree(OUT)
for a in POS:
    for s in ("train","val"):
        for c in ("present","absent"): (OUT/a/s/c).mkdir(parents=True)

n = collections.Counter()
for split in ("train","val"):
    for img_p in sorted((SRC/"images"/split).iterdir()):
        if img_p.suffix.lower() not in (".jpg",".jpeg",".png"): continue
        bgr = cv2.imread(str(img_p)); H,W = bgr.shape[:2]
        gt = collections.defaultdict(list)
        lab = SRC/"labels"/split/(img_p.stem+".txt")
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            f = line.split()
            if len(f) < 5: continue
            c=int(f[0]); cx,cy,bw,bh=(float(v) for v in f[1:5])
            gt[c].append([(cx-bw/2)*W,(cy-bh/2)*H,bw*W,bh*H])

        r = pose.predict(bgr, imgsz=640, conf=0.25, verbose=False, device=0)[0]
        if r.keypoints is None or not len(r.boxes): continue
        kp = r.keypoints.data.cpu().numpy()
        people = build_rois(r.boxes.xyxy.cpu().numpy(), kp)
        frs = [frontality(k) for k in kp]

        for i,p in enumerate(people):
            if p.head is not None and not p.head_collision:
                c = crop_of(bgr, p.head, GEOMETRY.head_crop_px)
                if c is not None:
                    for attr in ("head_cap","mask"):
                        if attr=="mask" and frs[i] < 0.5: n["mask_not_frontal"] += 1; continue
                        hp = any(inside(b, p.head) for b in gt[POS[attr]])
                        hn = any(inside(b, p.head) for b in gt[NEG[attr]])
                        if hp == hn: n[f"{attr}_no_evidence"] += 1; continue
                        cls = "present" if hp else "absent"
                        cv2.imwrite(str(OUT/attr/split/cls/f"{img_p.stem}_{i}.jpg"), c)
                        n[f"{attr}_{split}_{cls}"] += 1
            for hi,hb in enumerate(p.hands[:2]):
                c = crop_of(bgr, hb, GEOMETRY.hand_crop_px)
                if c is None: continue
                hp = any(inside(b, hb) for b in gt[POS["gloves"]])
                hn = any(inside(b, hb) for b in gt[NEG["gloves"]])
                if hp == hn: n["gloves_no_evidence"] += 1; continue
                cls = "present" if hp else "absent"
                cv2.imwrite(str(OUT/"gloves"/split/cls/f"{img_p.stem}_{i}_{hi}.jpg"), c)
                n[f"gloves_{split}_{cls}"] += 1

print(f"{'attribute':<12}{'train +/-':>14}{'val +/-':>12}")
for a in POS:
    tr = f"{n[a+'_train_present']}/{n[a+'_train_absent']}"
    va = f"{n[a+'_val_present']}/{n[a+'_val_absent']}"
    print(f"  {a:<10}{tr:>14}{va:>12}")
skipped = {k: v for k, v in n.items() if "evidence" in k or "frontal" in k}
print(f"\nskipped: {skipped}")
print(f"wrote {OUT}")
