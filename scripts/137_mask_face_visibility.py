"""Does v3's mask judgement depend on whether the face is visible at all?

For every 16 Sep device event: yolo11s-pose on the snapshot, match the device
person box, then ask the deployed facing test (nose keypoint >= 0.5 and the head
not turned away) -- the same gate the packed pipeline uses to decide whether a
mask can be judged. Cross-tab against v3's mask verdict.
"""
import collections, csv, json, sqlite3, sys
from pathlib import Path
import cv2, numpy as np
R = Path("/home/ubuntu/tryan/apd")
sys.path[:0] = [str(R), str(R/"deploy/apd_person")]
from apd_perception_ours import _facing_camera
from ultralytics import YOLO
A = R/"artifacts/replay16_person/prod_device"
db = sqlite3.connect(f"file:{A}/apd_storage/apd_events.db?mode=ro", uri=True)
box = {e: (json.loads(b), ip) for e, b, ip in db.execute(
    "select event_id,person_bbox,image_path from events where timestamp like '2026-09-16%'")}
rows = list(csv.DictReader(open(R/"artifacts/filter16/index.csv")))
pose = YOLO(str(R/"yolo11s-pose.pt"))


def iou(p, q):
    ix = max(0, min(p[2], q[2])-max(p[0], q[0])); iy = max(0, min(p[3], q[3])-max(p[1], q[1])); i = ix*iy
    return i/max(1e-6, (p[2]-p[0])*(p[3]-p[1])+(q[2]-q[0])*(q[3]-q[1])-i)


n = collections.Counter(); out = []
for j, r in enumerate(rows):
    b, ip = box[r["event_id"]]
    im = cv2.imread(str(A/ip))
    if im is None: continue
    pr = pose.predict(im, imgsz=640, conf=0.40, verbose=False)[0]
    face = "no_person"
    if len(pr.boxes):
        B = pr.boxes.xyxy.cpu().numpy(); K = pr.keypoints.data.cpu().numpy()
        k = int(np.argmax([iou(bb, b) for bb in B]))
        if iou(B[k], b) >= 0.4:
            face = "facing" if _facing_camera(K[k], 0.5, True) else "away"
    n[(face, r["ours_mask"])] += 1
    out.append({"event_id": r["event_id"], "face": face, "ours_mask": r["ours_mask"],
                "p_mask": r["p_mask"], "known_mask": r["known_mask"]})
    if j % 500 == 0: print(f"  {j}/{len(rows)}", flush=True)
with open(R/"artifacts/mask_check/face_visible.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("\nrows: face visibility (pose) x v3 mask verdict")
print(f"{'':10} {'violation':>10} {'ok':>6} {'unknown':>9}   total")
for f in ("facing", "away", "no_person"):
    t = sum(n[(f, v)] for v in ("violation", "ok", "unknown"))
    if not t: continue
    print(f"{f:10} {n[(f,'violation')]:10d} {n[(f,'ok')]:6d} {n[(f,'unknown')]:9d}   {t}")
