"""What to actually ship: which parts of the teacher can be dropped or shrunk.

The distillation targeted the wrong cost. Three yolo26n-cls classifiers are
4.6M parameters; DINOv2-base is 86.6M and yolo11x-pose is 58.8M. So the
question is not "can a small student replace the classifiers" -- they are
already small and already accurate -- but "how much of the anchor and the
second voter can be given up, and what does it cost".

Measured on the production annotation, which neither classifier trained on.
"""
import argparse, collections, json, pickle, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd.pose import build_rois
from apd.schema import GEOMETRY, assert_contract

assert_contract()
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ATT = ["head_cap", "mask", "gloves"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dump/dataset/apd_v3/dataset")
ap.add_argument("--poses", default="yolo11n-pose,yolo11s-pose,yolo11m-pose,yolo11x-pose")
ap.add_argument("--imgsz", type=int, default=1280)
a = ap.parse_args()

root = Path(a.data)
imgs = sorted(p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
y26 = {k: YOLO(f"runs/classify/runs/classify/runs/ci3_{k}/weights/best.pt") for k in ATT}
y26i = {k: {v: i for i, v in m.names.items()} for k, m in y26.items()}

def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

def centre_in(b, p):
    cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

gt = {}
for p in imgs:
    im = cv2.imread(str(p)); H,W = im.shape[:2]
    lp = root/"labels/val"/(p.stem+".txt"); g = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) < 5: continue
            c = int(f[0]); cx,cy,bw,bh = (float(v) for v in f[1:5])
            g.append(([(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H], NAMES[c]))
    gt[p] = g

print(f"{'pose model':16} {'params':>8} {'persons':>8} {'fps':>6}   " +
      "  ".join(f"{at:>10}" for at in ATT))
for tag in a.poses.split(","):
    w = f".work/{tag}.pt"
    if not Path(w).exists():
        YOLO(f"{tag}.pt").save(w) if False else None
        try: pose = YOLO(f"{tag}.pt")
        except Exception as e: print(f"{tag:16} unavailable ({e})"); continue
    else:
        pose = YOLO(w)
    nparam = sum(x.numel() for x in pose.model.parameters())
    stat = collections.defaultdict(collections.Counter); npers = 0
    t0 = time.time()
    for p in imgs:
        im = cv2.imread(str(p))
        rr = pose.predict(im, imgsz=a.imgsz, conf=0.40, verbose=False, device=0)[0]
        if rr.keypoints is None or not len(rr.boxes): continue
        people = build_rois(rr.boxes.xyxy.cpu().numpy(), rr.keypoints.data.cpu().numpy())
        npers += len(people)
        for pp in people:
            hc = crop_np(im, pp.head, GEOMETRY.head_crop_px) if (pp.head is not None and not pp.head_collision) else None
            hands = [c_ for c_ in (crop_np(im, h, GEOMETRY.hand_crop_px) for h in pp.hands[:2]) if c_ is not None]
            for at in ATT:
                hp = any(nm==at and centre_in(bb, pp.person_box) for bb,nm in gt[p])
                hn = any(nm==f"no_{at}" and centre_in(bb, pp.person_box) for bb,nm in gt[p])
                if hp == hn: continue
                true = "present" if hp else "absent"
                if at == "gloves":
                    if len(hands) < 2: continue
                    ps = [float(y26["gloves"].predict(c_, verbose=False, device=0)[0]
                          .probs.data[y26i["gloves"]["present"]]) for c_ in hands[:2]]
                    v = "present" if all(x >= .5 for x in ps) else "absent"
                else:
                    if hc is None: continue
                    q = float(y26[at].predict(hc, verbose=False, device=0)[0].probs.data[y26i[at]["present"]])
                    v = "present" if q >= .5 else "absent"
                stat[at]["n"] += 1; stat[at]["ok"] += int(v == true)
    dt = time.time() - t0
    accs = "  ".join(f"{stat[at]['ok']/max(1,stat[at]['n']):9.1%}" for at in ATT)
    print(f"{tag:16} {nparam/1e6:7.1f}M {npers:8d} {len(imgs)/dt:6.1f}   {accs}")
print("\n(voter4 only -- no DINOv2, no second voter; accuracy is over instances "
      "with a usable ROI and a ground-truth label)")
