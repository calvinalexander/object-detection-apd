"""Which gate actually causes each abstention, measured on a video."""
import argparse, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from ultralytics import YOLO
from apd.pose import build_rois, frontality
from apd.schema import GEOMETRY

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--stride", type=int, default=6)
args = ap.parse_args()
pose = YOLO(".work/yolo11x-pose.pt")
cap = cv2.VideoCapture(args.video)
R = collections.Counter(); n = 0; total_p = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    if n % args.stride: n += 1; continue
    r = pose.predict(fr, imgsz=GEOMETRY.pose_imgsz, conf=0.35, verbose=False, device=0)[0]
    if r.keypoints is not None and len(r.boxes):
        kp = r.keypoints.data.cpu().numpy()
        for p, k in zip(build_rois(r.boxes.xyxy.cpu().numpy(), kp), kp):
            total_p += 1
            # head_cap
            if p.head is None:          R["cap: no head ROI (no face kpts, no shoulders)"] += 1
            elif p.head_collision:      R["cap: neighbour's head inside the ROI"] += 1
            else:                       R["cap: DECIDED"] += 1
            # mask -- same gates plus geometric frontality
            f = frontality(k)
            if p.head is None:          R["mask: no head ROI"] += 1
            elif p.head_collision:      R["mask: neighbour's head inside the ROI"] += 1
            elif f < 0.5:               R["mask: not facing camera (frontality < 0.5)"] += 1
            else:                       R["mask: DECIDED"] += 1
            # gloves -- needs two hands to establish "both gloved"
            nh = len(p.hands)
            if nh == 0:                 R["gloves: no wrist visible"] += 1
            elif nh == 1:               R["gloves: only one wrist visible"] += 1
            else:                       R["gloves: DECIDED"] += 1
    n += 1
cap.release()
print(f"{total_p} person instances over {n//args.stride} frames\n")
for a in ("cap","mask","gloves"):
    rows = [(k,v) for k,v in R.items() if k.startswith(a+":")]
    tot = sum(v for _,v in rows)
    print(f"{a}")
    for k,v in sorted(rows, key=lambda x:-x[1]):
        print(f"   {k.split(': ',1)[1]:<46}{v:>7}{100*v/tot:>7.0f}%")
    print()
