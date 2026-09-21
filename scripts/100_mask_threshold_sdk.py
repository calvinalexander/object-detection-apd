"""Justify the mask threshold on enough "mask on" examples.

The COCO human labels hold only 6 people with a mask on, so the 0.5 -> 0.8 move
rested on almost nothing. Production's annotated validation set (SDK apd_v3 val)
has hundreds of annotator-drawn mask / no_mask boxes. Frames are kept only if
neither they nor any other frame of the same video fed the deployed mask trunk
(data/crops_cocoheld), since adjacent frames of one video are near-duplicates.

Each mask / no_mask box is matched to the pose person whose head ROI contains
its centre, the same way the device sees people, and the trunk's probability
is recorded. Boxes no pose person claims are counted, not guessed.
"""
import collections, json, re, sys
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd.pose import build_rois
from apd.schema import GEOMETRY
from apd_perception_ours import OursPerception, _crop, _stack
from ultralytics import YOLO

N = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
OUT = ROOT/"artifacts/error_analysis/mask_sdk"; OUT.mkdir(parents=True, exist_ok=True)


def stem(n):
    s = re.sub(r"__[a-z0-9]+\.jpg$", "", n); s = re.sub(r"\.jpg$", "", s); s = re.sub(r"^sdk_", "", s)
    m = re.match(r"(.+?)_(\d+)(?:_h\d+)?$", s); return m.group(1) if m else s


def video(f):
    m = re.match(r"([0-9a-f]{8})-", f); return m.group(1) if m else f


train_frames = {stem(p.name) for p in (ROOT/"data/crops_cocoheld/mask").rglob("*.jpg")}
train_videos = {video(f) for f in train_frames}
root = ROOT/"artifacts/sdk/sdk/customers/apd_v3/dataset"
frames, drop_frame, drop_video = [], 0, 0
for lp in sorted((root/"labels/val").glob("*.txt")):
    cl = [N[int(l.split()[0])] for l in lp.read_text().splitlines() if l.strip()]
    if "mask" not in cl and "no_mask" not in cl: continue
    if lp.stem in train_frames: drop_frame += 1; continue
    if video(lp.stem) in train_videos: drop_video += 1; continue
    frames.append(lp)
print(f"{len(frames)} independent frames (dropped {drop_frame} used in training, "
      f"{drop_video} from a video that was)")

clf = OursPerception(SimpleNamespace(backend="axelera", gate=1))
pose = YOLO(str(ROOT/"yolo11n-pose.pt"))
rows = []                           # (truth 1=violation, p_mask or None, face px)
TH = clf.thresh["mask"]
for d in OUT.iterdir():
    if d.is_dir(): __import__("shutil").rmtree(d)
counts = collections.Counter()


def save_face(im, cx, cy, sz, y, p, stem_, k):
    """Face with context, labelled, into mask_sdk/<outcome>/ at the deployed threshold."""
    t = "violation" if y else "ok"
    if p is None: out = "not_located"
    else:
        pred = "ok" if p >= TH else "violation"
        out = f"correct_{t}" if pred == t else f"WRONG_said_{pred}_truth_{t}"
    counts[out] += 1
    H, W = im.shape[:2]; r = max(24, int(sz*2.2))
    c = im[max(0, int(cy-r)):min(H, int(cy+r)), max(0, int(cx-r)):min(W, int(cx+r))].copy()
    if c.size == 0: return
    c = cv2.resize(c, (200, 200))
    x0, y0 = int(cx-r), int(cy-r); s_ = 200/(2*r)
    cv2.rectangle(c, (int((cx-sz/2-x0)*s_), int((cy-sz/2-y0)*s_)), (int((cx+sz/2-x0)*s_), int((cy+sz/2-y0)*s_)),
                  (60, 200, 60) if y == 0 else (60, 60, 235), 1)
    bar = np.full((34, 200, 3), 25, np.uint8)
    cv2.putText(bar, f"truth {'mask on' if y == 0 else 'no mask'}  {sz:.0f}px", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, .38, (230, 230, 230), 1)
    cv2.putText(bar, "not located by pose" if p is None else f"p(mask) {p:.2f} -> {'ok' if p >= TH else 'violation'}",
                (3, 28), cv2.FONT_HERSHEY_SIMPLEX, .38, (160, 220, 160), 1)
    d = OUT/out; d.mkdir(exist_ok=True)
    cv2.imwrite(str(d/f"{stem_}_{k}.jpg"), np.vstack([bar, c]))
for lp in frames:
    im = cv2.imread(str(root/"images/val"/(lp.stem+".jpg")))
    if im is None: continue
    H, W = im.shape[:2]
    faces = []
    for l in lp.read_text().splitlines():
        f = l.split()
        if len(f) < 5 or N[int(f[0])] not in ("mask", "no_mask"): continue
        cx, cy, bw, bh = [float(v) for v in f[1:5]]
        faces.append((1 if N[int(f[0])] == "no_mask" else 0, cx*W, cy*H, (bw*W*bh*H)**.5))
    r = pose.predict(im, imgsz=640, conf=0.40, verbose=False)[0]
    people = build_rois(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()) if len(r.boxes) else []
    heads = [(i, p.head) for i, p in enumerate(people) if p.head is not None and not p.head_collision]
    used = set()
    for k, (y, cx, cy, sz) in enumerate(faces):
        hit = None
        for i, h in heads:
            if i in used: continue
            if h[0] <= cx <= h[2] and h[1] <= cy <= h[3]:
                hit = h; used.add(i); break
        if hit is None:
            rows.append((y, None, sz)); save_face(im, cx, cy, sz, y, None, lp.stem[:40], k); continue
        c = _crop(im, hit, GEOMETRY.head_crop_px)
        p = None if c is None else float(clf.packed.run(None, {"head_crops": _stack([c]), "hand_crops": _stack([])})[1][0])
        rows.append((y, p, sz)); save_face(im, cx, cy, sz, y, p, lp.stem[:40], k)

Y = np.array([r[0] for r in rows]); judged = np.array([r[1] is not None for r in rows])
P = np.array([r[1] if r[1] is not None else -1 for r in rows])
yj, pj = Y[judged], P[judged]
L = ["# Mask threshold on production's annotated validation set", "",
     f"{len(frames)} frames never seen by the mask trunk, neither the frame nor its video. "
     f"{len(Y)} annotated faces: **{int((Y==0).sum())} mask on**, {int((Y==1).sum())} no mask. "
     f"Pose (yolo11n, 0.40) located the head of {judged.sum()} of them "
     f"({int((yj==0).sum())} mask on, {int((yj==1).sum())} no mask); the rest are counted, not scored.", "",
     "Compare the COCO human labels, which have 6 people with a mask on.", "",
     "## Calibration: what the trunk's probability means", "",
     "| p(mask) | faces | truly mask on | hard call at 0.5 right |", "|---|---|---|---|"]
for lo, hi in ((0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, .8), (.8, .9), (.9, 1.01)):
    m = (pj >= lo) & (pj < hi)
    if not m.any(): continue
    L.append(f"| {lo:.2f}-{min(hi,1):.2f} | {m.sum()} | {np.mean(yj[m]==0):.0%} | {np.mean((pj[m]>=.5)==(yj[m]==0)):.0%} |")
L += ["", "## Threshold sweep ('ok' only when p(mask) >= threshold; violation is positive)", "",
      "| threshold | precision | recall | MCC | mask-on people wrongly flagged | violations missed |",
      "|---|---|---|---|---|---|"]
best = None
for t in (.3, .4, .5, .6, .7, .8, .85, .9, .95):
    q = (pj < t).astype(int)
    mcc = matthews_corrcoef(yj, q) if len(set(q.tolist())) > 1 else 0.0
    fp = int(((q == 1) & (yj == 0)).sum()); fn = int(((q == 0) & (yj == 1)).sum())
    L.append(f"| {t:.2f} | {precision_score(yj,q,zero_division=0):.1%} | {recall_score(yj,q,zero_division=0):.1%} "
             f"| {mcc:.3f} | {fp} of {int((yj==0).sum())} | {fn} of {int((yj==1).sum())} |")
    if best is None or mcc > best[1]: best = (t, mcc)
L += ["", f"Best MCC here: threshold {best[0]:.2f} ({best[1]:.3f}).", ""]
L += [f"## Folders, at the deployed threshold {TH}", "",
      "Every annotated face is saved into one folder; the box is the annotator's face box "
      "(green = mask on, red = no mask).", "", "| folder | faces |", "|---|---|"]
for o, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    L.append(f"| `mask_sdk/{o}/` | {v} |")
L.append("")
small = judged & (np.array([r[2] for r in rows]) < 30)
L += [f"Faces under 30 px: {small.sum()} of {judged.sum()} judged."]
L += ["", (ROOT/"scripts/100_mask_threshold_sdk_notes.md").read_text()]
(OUT/"README.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
