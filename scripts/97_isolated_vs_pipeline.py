"""The same classifiers scored two ways: handed the right crop, or deployed.

Section 4 reported head_cap 0.523. Section 6d reported 0.355 for what looks
like the same models. Both are correct and they measure different things:

  isolated  the stored head/hand ROI for each labelled person is cropped and
            fed straight to the trunks -- how good is the classifier?
  pipeline  pose finds the people, build_rois derives the ROIs, the gate may
            abstain, and the verdict is matched back to the labelled person by
            overlap -- how good is the system that is deployed?

The gap between them is the cost of everything around the model.
"""
import collections, json, sys
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd.schema import GEOMETRY
from apd_perception_ours import OursPerception, _crop, _prep, _stack

ATTRS = ["head_cap", "mask", "gloves"]
GRP = {"head_cap": "headcap", "mask": "mask", "gloves": "gloves"}
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--gate", type=int, default=1); a = ap.parse_args()
clf = OursPerception(SimpleNamespace(backend="ultralytics", pose_weights=str(ROOT/"yolo11n-pose.pt"),
                                     person_conf=0.25, gate=a.gate, imgsz=640))
print(f"ROI gate: {'on' if a.gate else 'OFF'}")

persons = {json.loads(l)["person_id"]: json.loads(l) for l in (ROOT/"artifacts/persons.jsonl").read_text().splitlines()}
truth = collections.defaultdict(dict)
for line in (ROOT/"artifacts/seed_labels.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["person_id"] in persons:
        for at, v in (r.get("labels") or {}).items():
            if v in ("present", "absent"): truth[at][r["person_id"]] = v


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix*iy
    return i / max(1e-6, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def run_packed(head_crops, hand_crops):
    return clf.packed.run(None, {"head_crops": _stack(head_crops), "hand_crops": _stack(hand_crops)})


ys = {a: [] for a in ATTRS}
pi = {a: [] for a in ATTRS}     # isolated
pp = {a: [] for a in ATTRS}     # pipeline
abstain = collections.Counter()
frames = sorted({persons[p]["frame"] for a in ATTRS for p in truth[a]})
for k, fname in enumerate(frames, 1):
    im = cv2.imread(str(ROOT/"data/sppg"/fname))
    if im is None: continue
    cb, ck = clf.people(im, {}, "x")
    cboxes, cverd = clf.perceive(im, cb, ck)
    for at in ATTRS:
        for pid, tv in truth[at].items():
            rec = persons[pid]
            if rec["frame"] != fname: continue
            ys[at].append(0 if tv == "present" else 1)

            # --- isolated: the stored ROI, straight into the trunks
            if at == "gloves":
                hs = [c for c in (_crop(im, h, GEOMETRY.hand_crop_px) for h in (rec.get("hand_rois") or [])[:2]) if c is not None]
                if len(hs) < 2: pi[at].append(0); abstain[f"isolated/{at}"] += 1
                else:
                    g = run_packed([], hs[:2])[2]
                    pi[at].append(0 if float(min(g)) >= 0.5 else 1)
            else:
                hc = _crop(im, rec["head_roi"], GEOMETRY.head_crop_px) if rec.get("head_roi") else None
                if hc is None: pi[at].append(0); abstain[f"isolated/{at}"] += 1
                else:
                    o = run_packed([hc], [])
                    prob = float(o[0][0] if at == "head_cap" else o[1][0])
                    pi[at].append(0 if prob >= 0.5 else 1)

            # --- pipeline: pose, gate, match, exactly as deployed
            best, bi = 0.0, -1
            for i, cbx in enumerate(cboxes):
                v = iou(cbx, rec["person_box"])
                if v > best: best, bi = v, i
            d = cverd[bi]["apd"] if (bi >= 0 and best >= 0.3) else {}
            st = d.get(GRP[at], "unknown")
            if st not in ("ok", "violation"): abstain[f"pipeline/{at}"] += 1
            pp[at].append(1 if st == "violation" else 0)
    if k % 15 == 0: print(f"  {k}/{len(frames)}", flush=True)

print("\n=== the same classifiers, scored two ways (violation is positive) ===")
print(f"{'attribute':10} {'scoring':10} {'prec':>7} {'recall':>7} {'F1':>7} {'MCC':>7}   abstained")
for at in ATTRS:
    y = np.array(ys[at])
    for tag, pred in (("isolated", pi[at]), ("pipeline", pp[at])):
        q = np.array(pred)
        mcc = matthews_corrcoef(y, q) if len(set(q.tolist())) > 1 else 0.0
        print(f"{at:10} {tag:10} {precision_score(y,q,zero_division=0):7.1%} {recall_score(y,q,zero_division=0):7.1%} "
              f"{f1_score(y,q,zero_division=0):7.3f} {mcc:7.3f}   {abstain[f'{tag}/{at}']:4d}/{len(y)}")
