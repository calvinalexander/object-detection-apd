"""Every model on the same validation crops: detectors, classifiers, production.

The det7 val split is 401 person crops with attribute boxes. A box is a verdict
once you ask "is head_cap or no_head_cap present on this person", so the same
images can score a detector, a classifier and production side by side.

Each model is given the same crop:
    detectors    read their own classes off it
    classifiers  pose the crop, cut head and hand ROIs, run the packed trunks
    production   run the 7-class detector on it and read the classes

Scoring is operational, as everywhere else in this project: violation is the
positive class and an abstention is silence, so it counts as no alert.

One caveat is reported rather than hidden: 180 of the 401 crops come from
production snapshots whose boxes production itself drew, so the "all" block
flatters it. The native block is the clean comparison.
"""
import collections, json, sys
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
ATTRS = ["head_cap", "mask", "gloves"]
GRP = {"head_cap": "headcap", "mask": "mask", "gloves": "gloves"}

import openvino as ov
from ultralytics import YOLO
from apd_perception_ours import OursPerception

DET = {"yolo26n (det7)": ROOT/"runs/det7/yolo26n/weights/best.pt",
       "yolo26s (det7)": ROOT/"runs/det7/yolo26s/weights/best.pt"}
PROD = {"apd_mbg Dec-2025 (prod)": ROOT/"model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml",
        "apd_mbg Mar-2026 (new prod)": ROOT/"artifacts/sdk/sdk/build/apd_mbg_openvino_model/best.xml"}

dets = {k: YOLO(str(v)) for k, v in DET.items() if v.exists()}
core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
prods = {k: core.compile_model(core.read_model(str(v)), dev) for k, v in PROD.items() if v.exists()}
clf = OursPerception(SimpleNamespace(backend="ultralytics", pose_weights=str(ROOT/"yolo11n-pose.pt"),
                                     person_conf=0.25, gate=1, imgsz=640))
print(f"detectors {list(dets)}\nproduction {list(prods)}\nclassifiers: packed trunks "
      f"({'packed' if clf.packed is not None else 'two files'})\n")


def letterbox(img, s=640):
    h, w = img.shape[:2]; r = min(s/h, s/w); nh, nw = round(h*r), round(w*r)
    o = np.full((s, s, 3), 114, np.uint8); t, l = (s-nh)//2, (s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img, (nw, nh)); return o


def prod_labels(net, im, conf=0.30):
    img = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    o = net([(img.astype(np.float32)/255).transpose(2, 0, 1)[None]])[net.outputs[0]][0]
    if o.shape[0] < o.shape[1]: o = o.T
    sc = o[:, 4:11]; cl = sc.argmax(1); cf = sc.max(1); m = cf >= conf
    return {NAMES[c] for c in cl[m]}


def op(pos, neg):
    return 0 if pos == neg else (0 if pos else 1)


sites = {json.loads(l)["id"] for l in open(ROOT/"artifacts/consensus.jsonl")}
val = sorted((ROOT/"data/det7/images/val").iterdir())
ys = {a: {"all": [], "native": []} for a in ATTRS}
pr = {a: collections.defaultdict(lambda: {"all": [], "native": []}) for a in ATTRS}

for n, p in enumerate(val, 1):
    im = cv2.imread(str(p))
    if im is None: continue
    lab = {NAMES[int(l.split()[0])] for l in (ROOT/"data/det7/labels/val"/f"{p.stem}.txt").read_text().splitlines() if l.strip()}
    scope = ["all"] + ([] if p.stem.rsplit("_p", 1)[0] in sites else ["native"])

    det_lab = {k: ({NAMES[int(c)] for c in r.boxes.cls.cpu().numpy()} if len(r.boxes) else set())
               for k, r in ((k, m.predict(im, imgsz=416, conf=0.25, verbose=False)[0]) for k, m in dets.items())}
    prod_lab = {k: prod_labels(net, im) for k, net in prods.items()}
    boxes, kpts = clf.people(im, {}, "x")
    _, verdicts = clf.perceive(im, boxes, kpts)
    cv_ = verdicts[0]["apd"] if verdicts else {}

    for a in ATTRS:
        has_pos, has_neg = a in lab, f"no_{a}" in lab
        if has_pos == has_neg: continue                    # not labelled for this person
        y = 0 if has_pos else 1
        for s in scope:
            ys[a][s].append(y)
            for k, L in det_lab.items(): pr[a][k][s].append(op(a in L, f"no_{a}" in L))
            for k, L in prod_lab.items(): pr[a][k][s].append(op(a in L, f"no_{a}" in L))
            v = cv_.get(GRP[a], "unknown")
            pr[a]["classifiers (packed)"][s].append(1 if v == "violation" else 0)
    if n % 80 == 0: print(f"  {n}/{len(val)}", flush=True)

for scope, title in (("all", "all 401 val crops -- 180 carry boxes production drew, so it is flattered here"),
                     ("native", "native crops only -- boxes drawn by annotators, nobody has an advantage")):
    print(f"\n=== {title} ===")
    for a in ATTRS:
        y = np.array(ys[a][scope])
        if len(y) < 10 or len(set(y.tolist())) < 2:
            print(f"\n{a}: too few labels in this scope ({len(y)})"); continue
        print(f"\n{a}   n={len(y)}   violations={int(y.sum())} ({y.mean():.0%})")
        print(f"  {'model':28} {'prec':>7} {'recall':>7} {'F1':>7} {'MCC':>7} {'bal-acc':>8}")
        for name in ["always flag"] + list(pr[a]):
            q = np.ones_like(y) if name == "always flag" else np.array(pr[a][name][scope])
            print(f"  {name:28} {precision_score(y,q,zero_division=0):7.1%} {recall_score(y,q,zero_division=0):7.1%} "
                  f"{f1_score(y,q,zero_division=0):7.3f} "
                  f"{(matthews_corrcoef(y,q) if len(set(q.tolist()))>1 else 0.0):7.3f} "
                  f"{balanced_accuracy_score(y,q):8.1%}")
