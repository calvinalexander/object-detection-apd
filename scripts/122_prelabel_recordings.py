"""Model-assisted labelling for the DEVICE camera: people from data/recording.

Same method as scripts/119 (sppg), but the frames come from the recordings the
device itself produced, and the pre-labels come from the one-network person
model (runs/person_ppe/export). This is the site whose uniform - blue bouffant
cap, black skullcap, white gloves of the cleaning crew - no training set covers.

  - people found with yolo11s-pose @ 0.40 (the bundle's pose model)
  - person box at least 80 px tall
  - at most --per-clip people per clip, spread over all clips and over time
  - pre-labels: p(head_cap), p(mask), p(gloves) with the model's own thresholds

    python scripts/122_prelabel_recordings.py [--n 3000] [--per-clip 12]
Writes data/label_recording/{people.csv, person/<id>.jpg}; sheets come from scripts/120.
"""
import argparse, collections, csv, json, random, sys
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--per-clip", type=int, default=40)
ap.add_argument("--stride", type=float, default=2.0, help="seconds between sampled frames")
ap.add_argument("--min-h", type=int, default=80)
ap.add_argument("--out", default="data/label_recording")
a = ap.parse_args()
OUT = ROOT/a.out; (OUT/"person").mkdir(parents=True, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
meta = json.loads((ROOT/"runs/person_ppe/export/apd_ours_person_ppe.json").read_text())
TH = {"head_cap": meta["thresholds"]["head_cap"], "mask": meta["thresholds"]["mask"], "gloves": meta["thresholds"]["gloves"]}
sess = ort.InferenceSession(str(ROOT/"runs/person_ppe/export/apd_ours_person_ppe.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
pose = YOLO(str(ROOT/"yolo11s-pose.pt"))


def letterbox(im, s=320):
    h, w = im.shape[:2]; r = s/max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return o


clips = sorted(p for p in (ROOT/"data/recording").rglob("*.mp4") if p.stat().st_size > 1e6)
random.Random(7).shuffle(clips)
rows = []
for ci, clip in enumerate(clips):
    if len(rows) >= a.n: break
    cap = cv2.VideoCapture(str(clip)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); taken = 0
    recent = []                       # boxes already taken from this clip, to skip near-duplicates
    for sec in np.arange(2, max(3, n/fps), a.stride):
        if taken >= a.per_clip or len(rows) >= a.n: break
        cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000); ok, f = cap.read()
        if not ok: continue
        H, W = f.shape[:2]
        r = pose.predict(f, imgsz=640, conf=0.40, verbose=False)[0]
        if not len(r.boxes): continue
        B = r.boxes.xyxy.cpu().numpy(); K = r.keypoints.data.cpu().numpy()
        crops, keep = [], []
        for i, b in enumerate(B):
            x1, y1, x2, y2 = b; pw, ph = x2-x1, y2-y1
            if ph < a.min_h: continue
            c = f[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
            if c.size == 0 or min(c.shape[:2]) < 16: continue
            # keypoint sanity: a real person shows some upper-body joints
            vis = int((K[i][:11, 2] >= 0.35).sum())
            def iou(u, v):
                ix = max(0, min(u[2], v[2])-max(u[0], v[0])); iy = max(0, min(u[3], v[3])-max(u[1], v[1])); ii = ix*iy
                return ii/max(1e-6, (u[2]-u[0])*(u[3]-u[1])+(v[2]-v[0])*(v[3]-v[1])-ii)
            if any(iou(b, q) > 0.55 for q in recent): continue       # same person, barely moved
            recent.append(b)
            crops.append(c); keep.append((i, b, vis))
        if not crops: continue
        X = np.stack([((cv2.cvtColor(letterbox(c), cv2.COLOR_BGR2RGB).astype(np.float32)/255-MEAN)/STD).transpose(2, 0, 1) for c in crops])
        P = sess.run(None, {"person_crops": X})[0]
        for (i, b, vis), c, p in zip(keep, crops, P):
            if len(rows) >= a.n or taken >= a.per_clip: break
            pid = f"{clip.stem}_t{int(sec)}_p{i}"
            cv2.imwrite(str(OUT/"person"/f"{pid}.jpg"), c)
            rows.append({"id": pid, "clip": clip.stem, "sec": round(float(sec), 1),
                         "box": json.dumps([round(float(v), 1) for v in b]), "kp_visible": vis,
                         **{k: ("present" if float(p[j]) >= TH[k] else "absent") for j, k in enumerate(("head_cap", "mask", "gloves"))},
                         **{f"{k}_p": round(float(p[j]), 3) for j, k in enumerate(("head_cap", "mask", "gloves"))}})
            taken += 1
    if ci % 20 == 0: print(f"  clip {ci}/{len(clips)}, people {len(rows)}", flush=True)
for k, r in enumerate(rows): r["n"] = k
cols = ["n", "id", "clip", "sec", "box", "kp_visible", "head_cap", "head_cap_p", "mask", "mask_p", "gloves", "gloves_p"]
with open(OUT/"people.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c: r[c] for c in cols})
c = collections.Counter((k, r[k]) for r in rows for k in ("head_cap", "mask", "gloves"))
print(len(rows), "people from", len({r['clip'] for r in rows}), "clips;", dict(sorted(c.items())))
