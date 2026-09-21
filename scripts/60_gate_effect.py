"""Does the ROI gate improve correctness, or only reduce coverage?

Refusing to answer always raises accuracy-where-answered, so the gate has to be
judged on what it refuses. If the skipped ROIs were ones the classifier got
right, the gate is just throwing away coverage. Measured on the production
annotation, which the classifiers never trained on.
"""
import collections, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd.pose import build_rois, roi_usable
from apd.schema import GEOMETRY, assert_contract

assert_contract()
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ATT = ["head_cap", "mask", "gloves"]
root = Path("data/dump/dataset/apd_v3/dataset")
imgs = sorted(p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))

pose = YOLO(".work/yolo11x-pose.pt")
y26 = {k: YOLO(f"runs/classify/runs/classify/runs/ci3_{k}/weights/best.pt") for k in ATT}
y26i = {k: {v: i for i, v in m.names.items()} for k, m in y26.items()}

def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

def centre_in(b,p):
    cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

rows = []          # (attr, gated_ok, reason, correct)
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
    rr = pose.predict(im, imgsz=1280, conf=0.40, verbose=False, device=0)[0]
    if rr.keypoints is None or not len(rr.boxes): continue
    people = build_rois(rr.boxes.xyxy.cpu().numpy(), rr.keypoints.data.cpu().numpy())
    pboxes = [q.person_box for q in people]
    for pp in people:
        for at in ATT:
            hp = any(nm==at and centre_in(bb, pp.person_box) for bb,nm in gtb)
            hn = any(nm==f"no_{at}" and centre_in(bb, pp.person_box) for bb,nm in gtb)
            if hp == hn: continue
            true = "present" if hp else "absent"
            if at == "gloves":
                boxes = list(pp.hands[:2])
                checks = [roi_usable(h, im.shape, part="hand", others=pboxes,
                                     self_box=pp.person_box) for h in boxes]
                crops = [crop_np(im, h, GEOMETRY.hand_crop_px) for h in boxes]
                crops = [c for c in crops if c is not None]
                if len(crops) < 2: continue           # ungated pipeline also abstains
                ps = [float(y26["gloves"].predict(c, verbose=False, device=0)[0]
                      .probs.data[y26i["gloves"]["present"]]) for c in crops]
                v = "present" if all(x >= .5 for x in ps) else "absent"
                bad = [c for c in checks if not c]
                ok, reason = (not bad), (bad[0].reason if bad else "")
            else:
                ck = roi_usable(pp.head, im.shape, part="head", others=pboxes,
                                self_box=pp.person_box)
                hc = crop_np(im, pp.head, GEOMETRY.head_crop_px) if pp.head is not None else None
                if hc is None: continue
                q = float(y26[at].predict(hc, verbose=False, device=0)[0].probs.data[y26i[at]["present"]])
                v = "present" if q >= .5 else "absent"
                ok, reason = bool(ck), ck.reason
            rows.append((at, ok, reason, v == true))
    if n % 40 == 0: print(f"  {n}/{len(imgs)}", flush=True)

print(f"\n{'attribute':10} {'gate off':>22} {'gate on':>22} {'skipped':>20}")
for at in ATT:
    r = [x for x in rows if x[0] == at]
    if not r: continue
    all_acc = sum(x[3] for x in r) / len(r)
    kept = [x for x in r if x[1]]
    skip = [x for x in r if not x[1]]
    kacc = sum(x[3] for x in kept)/len(kept) if kept else float("nan")
    sacc = sum(x[3] for x in skip)/len(skip) if skip else float("nan")
    print(f"{at:10} {all_acc:8.1%} (n={len(r):4d}) {kacc:11.1%} (n={len(kept):4d}) "
          f"{sacc:9.1%} (n={len(skip):3d})")

print("\nwhy ROIs were skipped")
why = collections.Counter(x[2] for x in rows if not x[1])
for k, v in why.most_common():
    sub = [x for x in rows if x[2] == k]
    print(f"  {k:34} {v:4d}   would have been right {sum(x[3] for x in sub)/len(sub):.0%} of the time")
