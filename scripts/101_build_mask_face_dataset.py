"""Face-centred crops for a dedicated mask trunk.

Labels: annotator mask / no_mask face boxes from the four annotated sources.
Each box is matched to a pose person whose head region contains its centre,
and that person's face_roi() -- the same function the device will call -- is
cropped at several gains so the size can be chosen on validation.

Split by video, never by frame. TEST is drawn only from videos the currently
deployed mask trunk never saw (data/crops_cocoheld), so old and new trunks are
compared on ground neither trained on.

    train  crops from yolo11x AND yolo11n keypoints, 3 geometric jitters each
    val    yolo11n only, no jitter -- what the device produces
    test   yolo11n only, no jitter
"""
import collections, hashlib, json, re, shutil, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd.pose import build_rois, face_roi
from apd.schema import GEOMETRY
from apd_perception_ours import _crop
from ultralytics import YOLO

N = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
GAINS = [0.5, 0.8, 1.1, "head"]      # "head" = the shared cap-sized region, as a control
OUT = ROOT/"data/mask_face"
SOURCES = [ROOT/"artifacts/sdk/sdk/customers/apd_v3/dataset", ROOT/"data/dump/dataset/apd_v2/dataset",
           ROOT/"data/dump/dataset/apd_v3/dataset", ROOT/"data/voter3"]
rng = np.random.default_rng(20260917)


def stem_of_crop(n):
    s = re.sub(r"__[a-z0-9]+\.jpg$", "", n); s = re.sub(r"\.jpg$", "", s); s = re.sub(r"^sdk_", "", s)
    m = re.match(r"(.+?)_(\d+)(?:_h\d+)?$", s); return m.group(1) if m else s


def video(stem):
    m = re.match(r"([0-9a-f]{8})-", stem); return m.group(1) if m else stem


def bucket(key):                                   # stable 0..99
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100


old_train = {stem_of_crop(p.name) for p in (ROOT/"data/crops_cocoheld/mask").rglob("*.jpg")}
old_videos = {video(s) for s in old_train}

frames = {}                                        # stem -> (image path, faces)
for src in SOURCES:
    for lp in src.rglob("*.txt"):
        if "labels" not in lp.parts or lp.stem in frames: continue
        faces = []
        for l in lp.read_text().splitlines():
            f = l.split()
            if len(f) >= 5 and N[int(f[0])] in ("mask", "no_mask"):
                faces.append((N[int(f[0])] == "mask", *map(float, f[1:5])))
        if not faces: continue
        img = Path(str(lp.parent).replace("labels", "images"))/(lp.stem + ".jpg")
        if img.exists(): frames[lp.stem] = (img, faces)

split = {}
for s in frames:
    v = video(s)
    if v not in old_videos and bucket("test" + v) < 45: split[s] = "test"
    elif bucket("val" + v) < 15: split[s] = "val"
    else: split[s] = "train"
print(f"{len(frames)} annotated frames with a mask label: "
      + ", ".join(f"{k} {v}" for k, v in collections.Counter(split.values()).items()))
tv = {video(s) for s, k in split.items() if k == "test"}
assert not tv & {video(s) for s, k in split.items() if k != "test"}, "a test video leaked"
assert not tv & old_videos, "a test video was seen by the deployed trunk"

if OUT.exists(): shutil.rmtree(OUT)
poses = {"n": YOLO(str(ROOT/"yolo11n-pose.pt")), "x": YOLO(str(ROOT/"yolo11x-pose.pt"))}
count = collections.Counter(); manifest = []
for s, (img, faces) in sorted(frames.items()):
    im = cv2.imread(str(img))
    if im is None: continue
    H, W = im.shape[:2]
    which = ["n", "x"] if split[s] == "train" else ["n"]
    for pm in which:
        conf = 0.40 if pm == "n" else 0.25
        r = poses[pm].predict(im, imgsz=640 if pm == "n" else 960, conf=conf, verbose=False)[0]
        if not len(r.boxes): 
            for _ in faces: count[(split[s], pm, "not_located")] += 1
            continue
        boxes = r.boxes.xyxy.cpu().numpy(); K = r.keypoints.data.cpu().numpy()
        people = build_rois(boxes, K)
        used = set()
        for fi, (present, cx, cy, bw, bh) in enumerate(faces):
            cx, cy = cx*W, cy*H
            hit = None
            for i, p in enumerate(people):
                if i in used or p.head is None: continue
                h = p.head
                if h[0] <= cx <= h[2] and h[1] <= cy <= h[3]: hit = i; break
            if hit is None:
                count[(split[s], pm, "not_located")] += 1; continue
            used.add(hit)
            cls = "present" if present else "absent"
            jit = [(0, 0, 1.0)] if split[s] != "train" else \
                  [(0, 0, 1.0)] + [tuple(rng.uniform([-.1, -.1, .85], [.1, .1, 1.15])) for _ in range(2)]
            for g in GAINS:
                base = people[hit].head if g == "head" else face_roi(K[hit], tuple(boxes[hit]), gain=g)
                if base is None: continue
                bw_, bh_ = base[2] - base[0], base[3] - base[1]
                mx, my = (base[0]+base[2])/2, (base[1]+base[3])/2
                for j, (dx, dy, sc) in enumerate(jit):
                    hw, hh = bw_*sc/2, bh_*sc/2; x0, y0 = mx + dx*bw_, my + dy*bh_
                    c = _crop(im, (x0-hw, y0-hh, x0+hw, y0+hh), GEOMETRY.face_crop_px)
                    if c is None: continue
                    d = OUT/f"g{g}"/split[s]/cls; d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d/f"{s[:60]}_{fi}_{pm}{j}.jpg"), c)
            count[(split[s], pm, cls)] += 1
            if split[s] == "test" and pm == "n":
                manifest.append({"stem": s, "face": fi, "present": bool(present),
                                 "head": [float(v) for v in people[hit].head],
                                 "kp": K[hit].tolist(), "box": boxes[hit].tolist(), "img": str(img)})
(OUT/"test_manifest.json").write_text(json.dumps(manifest))
print(f"\n{'split':6} {'pose':4} {'mask on':>8} {'no mask':>8} {'not located':>12}")
for sp in ("train", "val", "test"):
    for pm in ("n", "x"):
        c = [count[(sp, pm, k)] for k in ("present", "absent", "not_located")]
        if sum(c): print(f"{sp:6} {pm:4} {c[0]:8d} {c[1]:8d} {c[2]:12d}")
print(f"\nwrote {OUT}")
