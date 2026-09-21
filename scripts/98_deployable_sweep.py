"""Which deployable configuration is actually best, measured end to end.

The benchmark numbers and the device disagree because the benchmark hands each
model the right crop, while on the device a pose model has to find the person
first. Section 6d measured that cost: yolo11n locates only 55% of the labelled
people, so half the classifier's opportunities never arrive.

This sweeps the three things that can actually be changed on the device and
scores every combination the same way: operational verdicts on the 53 COCO
human-labelled frames, violation positive, matched to the labelled person by
overlap. Nothing is handed a ground-truth box.

    pose model   yolo11n (compiled today) / yolo11s (would need a recompile)
    pose conf    conf_threshold in config.yaml, baked into the build
    classifier   packed trunks (deployed) / yolo26n crop / yolo26s crop
"""
import collections, itertools, json, sys
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd.pose import build_rois
from apd_perception_ours import OursPerception
from ultralytics import YOLO

ATTRS = ["head_cap", "mask", "gloves"]
GRP = {"head_cap": "headcap", "mask": "mask", "gloves": "gloves"}
NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
PAD = 0.18

POSES = {"yolo11n": ROOT/"yolo11n-pose.pt", "yolo11s": ROOT/"yolo11s-pose.pt"}
CONFS = [0.40, 0.25, 0.10]
DETS = {"yolo26n crop": ROOT/"runs/det7/yolo26n/weights/best.pt",
        "yolo26s crop": ROOT/"runs/det7/yolo26s/weights/best.pt"}

clf = OursPerception(SimpleNamespace(backend="axelera", gate=1))     # packed trunks, no pose of its own
dets = {k: YOLO(str(v)) for k, v in DETS.items() if v.exists()}
poses = {k: YOLO(str(v)) for k, v in POSES.items() if v.exists()}

persons = {json.loads(l)["person_id"]: json.loads(l) for l in (ROOT/"artifacts/persons.jsonl").read_text().splitlines()}
truth = collections.defaultdict(dict)
for line in (ROOT/"artifacts/seed_labels.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["person_id"] in persons:
        for at, v in (r.get("labels") or {}).items():
            if v in ("present", "absent"): truth[at][r["person_id"]] = v
frames = sorted({persons[p]["frame"] for a in ATTRS for p in truth[a]})
print(f"{len(frames)} frames, " + ", ".join(f"{a} {len(truth[a])}" for a in ATTRS) + " labelled people\n")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix*iy
    return i / max(1e-6, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


results = {}
for pname, conf in itertools.product(poses, CONFS):
    ys = {a: [] for a in ATTRS}
    preds = {a: collections.defaultdict(list) for a in ATTRS}
    found = miss = 0
    for fname in frames:
        im = cv2.imread(str(ROOT/"data/sppg"/fname))
        if im is None: continue
        H, W = im.shape[:2]
        r = poses[pname].predict(im, imgsz=640, conf=conf, verbose=False)[0]
        if len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy(); kpts = r.keypoints.data.cpu().numpy()
        else:
            boxes, kpts = np.zeros((0, 4)), np.zeros((0, 17, 3))
        pboxes, verdicts = clf.perceive(im, boxes, kpts)          # packed trunks on pose ROIs
        # crop detectors on the same pose person boxes
        det_v = {k: [] for k in dets}
        for pb in pboxes:
            pw, ph = pb[2]-pb[0], pb[3]-pb[1]
            x1, y1 = int(max(0, pb[0]-PAD*pw)), int(max(0, pb[1]-PAD*ph))
            x2, y2 = int(min(W, pb[2]+PAD*pw)), int(min(H, pb[3]+PAD*ph))
            crop = im[y1:y2, x1:x2]
            for k, m in dets.items():
                if crop.size == 0 or min(crop.shape[:2]) < 16: det_v[k].append(set()); continue
                rr = m.predict(crop, imgsz=416, conf=0.25, verbose=False)[0]
                det_v[k].append({NAMES[int(c)] for c in rr.boxes.cls.cpu().numpy()} if len(rr.boxes) else set())

        for at in ATTRS:
            for pid, tv in truth[at].items():
                rec = persons[pid]
                if rec["frame"] != fname: continue
                ys[at].append(0 if tv == "present" else 1)
                best, bi = 0.0, -1
                for i, pb in enumerate(pboxes):
                    v = iou(pb, rec["person_box"])
                    if v > best: best, bi = v, i
                ok = bi >= 0 and best >= 0.3
                if at == "head_cap": found += ok; miss += not ok
                st = verdicts[bi]["apd"].get(GRP[at], "unknown") if ok else "unknown"
                preds[at]["packed classifiers"].append(1 if st == "violation" else 0)
                for k in dets:
                    L = det_v[k][bi] if ok else set()
                    pos, neg = at in L, f"no_{at}" in L
                    preds[at][k].append(0 if pos == neg else (0 if pos else 1))
    for at in ATTRS:
        y = np.array(ys[at])
        for k, q in preds[at].items():
            q = np.array(q)
            results[(pname, conf, k, at)] = (
                precision_score(y, q, zero_division=0), recall_score(y, q, zero_division=0),
                matthews_corrcoef(y, q) if len(set(q.tolist())) > 1 else 0.0)
    print(f"{pname} conf {conf:.2f}: located {found}/{found+miss} labelled people ({found/max(1,found+miss):.0%})", flush=True)

print(f"\n{'pose':8} {'conf':>5} {'classifier':20} " + " ".join(f"{a+' MCC':>14}" for a in ATTRS))
best = {}
for (pname, conf, k) in sorted({(a, b, c) for a, b, c, _ in results}):
    row = [results[(pname, conf, k, a)] for a in ATTRS]
    print(f"{pname:8} {conf:5.2f} {k:20} " + " ".join(f"{m:14.3f}" for _, _, m in row))
    for a, (p, r, m) in zip(ATTRS, row):
        if m > best.get(a, (None, -9))[1]: best[a] = ((pname, conf, k, p, r), m)
print("\nbest deployable configuration per attribute:")
for a, ((pname, conf, k, p, r), m) in best.items():
    print(f"  {a:9} {pname} conf {conf:.2f} + {k:20} MCC {m:.3f}  (P {p:.1%}, R {r:.1%})")
print("\nfor reference, production on the same people: head_cap 0.191, mask 0.146, gloves 0.047")
