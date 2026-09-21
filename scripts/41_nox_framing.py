"""Compare two label framings for the crop classifiers.

  current   positive = the `x` box is in the ROI
            negative = the `no_x` box is in the ROI
            skipped  = neither (the annotator declined to call it)

  no_x      positive = the `no_x` box is in the ROI  -- a violation
            negative = everything else, including the crops the current framing
                       discards

The second matches the product question ("is this a violation?") and keeps the
crops the first throws away, but it pays for that by treating "not annotated" as
"compliant", which is the inference-from-absence problem that wrecked the SPPG
gloves labels. So it is worth measuring rather than reasoning about.
"""
import collections, shutil, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd.pose import build_rois, frontality
from apd.schema import GEOMETRY

POS={"head_cap":2,"mask":3,"gloves":1}; NEG={"head_cap":5,"mask":6,"gloves":4}
SRC=Path("data/voter3"); OUT=Path("data/crops_nox")
pose=YOLO(".work/yolo11x-pose.pt")
def crop_np(img,box,size):
    x1,y1,x2,y2=box; pad=0.35*max(x2-x1,y2-y1)
    c=img[max(0,int(y1-pad)):int(y2+pad),max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))
def inside(b,roi):
    cx,cy=b[0]+b[2]/2,b[1]+b[3]/2
    return roi[0]<=cx<=roi[2] and roi[1]<=cy<=roi[3]

if OUT.exists(): shutil.rmtree(OUT)
for a in POS:
    for s in ("train","val"):
        for c in ("violation","ok"): (OUT/a/s/c).mkdir(parents=True)
n=collections.Counter()
for split in ("train","val"):
    for img_p in sorted((SRC/"images"/split).iterdir()):
        if img_p.suffix.lower() not in (".jpg",".jpeg",".png"): continue
        bgr=cv2.imread(str(img_p)); H,W=bgr.shape[:2]
        gt=collections.defaultdict(list)
        lab=SRC/"labels"/split/(img_p.stem+".txt")
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            f=line.split()
            if len(f)<5: continue
            c=int(f[0]); cx,cy,bw,bh=(float(v) for v in f[1:5])
            gt[c].append([(cx-bw/2)*W,(cy-bh/2)*H,bw*W,bh*H])
        r=pose.predict(bgr,imgsz=640,conf=0.30,verbose=False,device=0)[0]
        if r.keypoints is None or not len(r.boxes): continue
        kp=r.keypoints.data.cpu().numpy()
        people=build_rois(r.boxes.xyxy.cpu().numpy(),kp); frs=[frontality(k) for k in kp]
        for i,p in enumerate(people):
            if p.head is not None and not p.head_collision:
                c=crop_np(bgr,p.head,GEOMETRY.head_crop_px)
                if c is not None:
                    for at in ("head_cap","mask"):
                        if at=="mask" and frs[i]<0.5: continue
                        cls="violation" if any(inside(b,p.head) for b in gt[NEG[at]]) else "ok"
                        cv2.imwrite(str(OUT/at/split/cls/f"{img_p.stem}_{i}.jpg"),c); n[f"{at}_{split}_{cls}"]+=1
            for hi,hb in enumerate(p.hands[:2]):
                c=crop_np(bgr,hb,GEOMETRY.hand_crop_px)
                if c is None: continue
                cls="violation" if any(inside(b,hb) for b in gt[NEG["gloves"]]) else "ok"
                cv2.imwrite(str(OUT/"gloves"/split/cls/f"{img_p.stem}_{i}_{hi}.jpg"),c); n[f"gloves_{split}_{cls}"]+=1
print(f"{'attribute':<11}{'train viol/ok':>18}{'val viol/ok':>16}")
for a in POS:
    tr=f"{n[a+'_train_violation']}/{n[a+'_train_ok']}"
    va=f"{n[a+'_val_violation']}/{n[a+'_val_ok']}"
    print(f"  {a:<9}{tr:>18}{va:>16}")
print(f"wrote {OUT}")
