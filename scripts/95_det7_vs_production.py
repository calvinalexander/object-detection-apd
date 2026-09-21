"""The YOLO26 attribute detectors against production, on human labels only.

The det7 validation split cannot settle this: 180 of its 401 crops carry boxes
the production model drew, so scoring production there is scoring it against
itself. The 53 COCO-annotated frames are human-labelled and were excluded from
training entirely, which makes them the only honest test.

Protocol is section 4's, unchanged, so the numbers sit beside the classifier
ones: operational scoring, where an abstention is silence and therefore no
alert, and violation is the positive class. Each model runs in its own domain,
production on the full frame as it is deployed, the detectors on the person
crop they were trained for.
"""
import collections, json, sys
from pathlib import Path
import cv2, numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)
from scipy.stats import chi2


def mcnemar_p(base_ok, other_ok):
    """Paired test on the same people: only the disagreements carry information."""
    b = int((base_ok & ~other_ok).sum()); c = int((~base_ok & other_ok).sum())
    if b + c == 0: return 1.0
    return float(chi2.sf((abs(b-c)-1)**2 / (b+c), 1))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ATTRS = ["head_cap", "mask", "gloves"]
NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
PAD = 0.18

import openvino as ov
from ultralytics import YOLO
from types import SimpleNamespace
sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd_perception_ours import OursPerception

PROD = {"apd_mbg Dec-2025 (prod)": ROOT/"model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml",
        "apd_mbg Mar-2026 (new prod)": ROOT/"artifacts/sdk/sdk/build/apd_mbg_openvino_model/best.xml"}
OURS = {"yolo26n crop": ROOT/"runs/det7/yolo26n/weights/best.pt",
        "yolo26s crop": ROOT/"runs/det7/yolo26s/weights/best.pt"}
# full-frame models run like production: on the whole frame, no person handed
# to them, verdict from attribute boxes falling inside the labelled person
FRAME = {"yolo26n full-frame": ROOT/"runs/det7f/yolo26n/weights/best.pt",
         "yolo26s full-frame": ROOT/"runs/det7f/yolo26s/weights/best.pt"}

core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
prod_nets = {k: core.compile_model(core.read_model(str(v)), dev) for k, v in PROD.items() if v.exists()}
our_nets = {k: YOLO(str(v)) for k, v in OURS.items() if v.exists()}
frame_nets = {k: YOLO(str(v)) for k, v in FRAME.items() if v.exists()}
GRP = {"head_cap": "headcap", "mask": "mask", "gloves": "gloves"}
# the classifiers get the FULL frame, which is the domain they were built for:
# their ROI gate rejects boxes touching the image edge, so a tight person crop
# would make them abstain on most heads and hands
clf = OursPerception(SimpleNamespace(backend="ultralytics", pose_weights=str(ROOT/"yolo11n-pose.pt"),
                                     person_conf=0.25, gate=1, imgsz=640))
print(f"production: {list(prod_nets)}\ndetectors: {list(our_nets)}\n"
      f"classifiers: packed trunks on the full frame\n")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix*iy
    return inter / max(1e-6, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def letterbox(img, s=640):
    h, w = img.shape[:2]; r = min(s/h, s/w); nh, nw = round(h*r), round(w*r)
    o = np.full((s, s, 3), 114, np.uint8); t, l = (s-nh)//2, (s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img, (nw, nh)); return o, r, l, t


def centre_in(b, p):
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]


def prod_detect(net, im, conf=0.30):
    img, r, dx, dy = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    port = net.outputs[0]
    o = net([(img.astype(np.float32)/255).transpose(2, 0, 1)[None]])[port][0]
    if o.shape[0] < o.shape[1]: o = o.T
    sc = o[:, 4:11]; cl = sc.argmax(1); cf = sc.max(1); m = cf >= conf
    b, cf, cl = o[m][:, :4], cf[m], cl[m]
    if not len(b): return []
    xy = np.stack([b[:, 0]-b[:, 2]/2, b[:, 1]-b[:, 3]/2, b[:, 0]+b[:, 2]/2, b[:, 1]+b[:, 3]/2], 1)
    xy[:, [0, 2]] -= dx; xy[:, [1, 3]] -= dy; xy /= r
    idx = cv2.dnn.NMSBoxes([[float(v[0]), float(v[1]), float(v[2]-v[0]), float(v[3]-v[1])] for v in xy],
                           cf.tolist(), conf, 0.7)
    idx = np.array(idx).flatten() if len(idx) else np.zeros(0, int)
    return [(xy[i], NAMES[cl[i]]) for i in idx if NAMES[cl[i]] != "person"]


def verdict(found_pos, found_neg):
    """Operational: 1 alerts, 0 stays silent. Neither box and both boxes are silence."""
    return 0 if found_pos == found_neg else (0 if found_pos else 1)


persons = {json.loads(l)["person_id"]: json.loads(l) for l in (ROOT/"artifacts/persons.jsonl").read_text().splitlines()}
truth = collections.defaultdict(dict)
for line in (ROOT/"artifacts/seed_labels.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["person_id"] not in persons: continue
    for at, v in (r.get("labels") or {}).items():
        if v in ("present", "absent"): truth[at][r["person_id"]] = v

frames = sorted({persons[p]["frame"] for at in ATTRS for p in truth[at]})
print(f"{len(frames)} human-annotated frames, "
      + ", ".join(f"{at} {len(truth[at])}" for at in ATTRS) + " labelled persons\n")

ys = {at: [] for at in ATTRS}
preds = {at: collections.defaultdict(list) for at in ATTRS}
for k, fname in enumerate(frames, 1):
    im = cv2.imread(str(ROOT/"data/sppg"/fname))
    if im is None: continue
    H, W = im.shape[:2]
    ppe_by = {n: prod_detect(net, im) for n, net in prod_nets.items()}
    for n, m in frame_nets.items():
        r0 = m.predict(im, imgsz=640, conf=0.25, verbose=False)[0]
        ppe_by[n] = [] if not len(r0.boxes) else [
            (b, NAMES[int(c)]) for b, c in zip(r0.boxes.xyxy.cpu().numpy(), r0.boxes.cls.cpu().numpy())
            if NAMES[int(c)] != "person"]
    cb, ck = clf.people(im, {}, "x")
    cboxes, cverd = clf.perceive(im, cb, ck)

    for at in ATTRS:
        for pid, tv in truth[at].items():
            rec = persons[pid]
            if rec["frame"] != fname: continue
            pb = rec["person_box"]
            ys[at].append(0 if tv == "present" else 1)

            for n, ppe in ppe_by.items():
                pos = any(nm == at and centre_in(bb, pb) for bb, nm in ppe)
                neg = any(nm == f"no_{at}" and centre_in(bb, pb) for bb, nm in ppe)
                preds[at][n].append(verdict(pos, neg))

            pw, ph = pb[2]-pb[0], pb[3]-pb[1]
            x1, y1 = int(max(0, pb[0]-PAD*pw)), int(max(0, pb[1]-PAD*ph))
            x2, y2 = int(min(W, pb[2]+PAD*pw)), int(min(H, pb[3]+PAD*ph))
            crop = im[y1:y2, x1:x2]
            best, bi = 0.0, -1
            for i, cbx in enumerate(cboxes):
                v = iou(cbx, pb)
                if v > best: best, bi = v, i
            cd = cverd[bi]["apd"] if (bi >= 0 and best >= 0.3) else {}
            preds[at]["classifiers (packed)"].append(1 if cd.get(GRP[at]) == "violation" else 0)

            for n, m in our_nets.items():
                if crop.size == 0 or min(crop.shape[:2]) < 16:
                    preds[at][n].append(0); continue
                r0 = m.predict(crop, imgsz=416, conf=0.25, verbose=False)[0]
                lab = [NAMES[int(c)] for c in r0.boxes.cls.cpu().numpy()] if len(r0.boxes) else []
                preds[at][n].append(verdict(at in lab, f"no_{at}" in lab))
    if k % 10 == 0: print(f"  {k}/{len(frames)}", flush=True)

print("\n=== operational scores on human labels (violation is positive) ===")
summary = {}
for at in ATTRS:
    y = np.array(ys[at])
    basename = next(iter(prod_nets))
    base = np.array(preds[at][basename])
    print(f"\n{at}   n={len(y)}   violations={int(y.sum())} ({y.mean():.0%})")
    print(f"  {'model':28} {'prec':>7} {'recall':>7} {'F1':>7} {'MCC':>7} {'bal-acc':>8} | {'p vs prod':>10}")
    rows = [("always flag", np.ones_like(y))] + [(n, np.array(preds[at][n])) for n in preds[at]]
    for name, p in rows:
        pr = precision_score(y, p, zero_division=0); rc = recall_score(y, p, zero_division=0)
        f1 = f1_score(y, p, zero_division=0)
        mcc = matthews_corrcoef(y, p) if len(set(p.tolist())) > 1 else 0.0
        ba = balanced_accuracy_score(y, p)
        if name == basename:
            tag = "baseline"
        else:
            tag = f"{mcnemar_p(base == y, p == y):.2e}"
        print(f"  {name:28} {pr:7.1%} {rc:7.1%} {f1:7.3f} {mcc:7.3f} {ba:8.1%} | {tag:>10}")
        summary[f"{at}/{name}"] = {"precision": float(pr), "recall": float(rc), "f1": float(f1),
                                   "mcc": float(mcc), "bal_acc": float(ba)}
out = ROOT/"artifacts/det7_vs_production.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\nwrote {out}")
