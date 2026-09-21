"""COCO test for the v2 person models (presence + judgeability), against v1 and the deployed pipeline.

Two things are measured on the same people, end to end (yolo11s-pose @0.40):
  judgeable people (human said present/absent) -> precision / recall / F1 / MCC, violation positive
  unjudgeable people (human said unclear)      -> how often a violation is still claimed
COCO is never used for training or threshold picking.

    python scripts/128_eval_person_v2_coco.py
"""
import collections, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
import cv2, numpy as np, torch, torchvision
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"deploy/apd_ours")]
from apd.pose import build_rois
from apd_perception_ours import OursPerception
from ultralytics import YOLO

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
ITEMS = ["head_cap", "mask", "gloves"]
GRP = {"head_cap": "headcap", "mask": "mask", "gloves": "gloves"}


def lb(im, s=320):
    h, w = im.shape[:2]; r = s/max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return ((cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)


def load(arch, w, n_out):
    net = getattr(torchvision.models, arch)(weights=None)
    if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, n_out)
    else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, n_out)
    net.load_state_dict(torch.load(w, map_location="cpu")); return net.eval().cuda()


models = {}
for sp in sorted((ROOT/"runs/person_ppe_v2").glob("*/summary.json")):
    j = json.loads(sp.read_text()); models[f"v2 {j['arch']}"] = ("v2", load(j["arch"], j["weights"], 6), j["cuts"])
j1 = json.loads((ROOT/"runs/person_ppe/efficientnet_b0_320_sppg/summary.json").read_text())
models["v1 efficientnet_b0"] = ("v1", load(j1["arch"], j1["weights"], 3),
                                {k: {"present_below": v["threshold"]} for k, v in j1["items"].items()})
dep = OursPerception(NS(backend="axelera", gate=1))

persons = {json.loads(l)["person_id"]: json.loads(l) for l in open(ROOT/"artifacts/persons.jsonl")}
lab = collections.defaultdict(dict)
for l in open(ROOT/"artifacts/seed_labels.jsonl"):
    r = json.loads(l)
    if r["person_id"] in persons:
        for at, v in (r.get("labels") or {}).items(): lab[at][r["person_id"]] = v
pose = YOLO(str(ROOT/"yolo11s-pose.pt"))


def iou(p, q):
    ix = max(0, min(p[2], q[2])-max(p[0], q[0])); iy = max(0, min(p[3], q[3])-max(p[1], q[1])); i = ix*iy
    return i/max(1e-6, (p[2]-p[0])*(p[3]-p[1])+(q[2]-q[0])*(q[3]-q[1])-i)


res = collections.defaultdict(lambda: collections.defaultdict(list))   # model -> attr -> (truth, alert)
for fn in sorted({persons[p]["frame"] for at in lab for p in lab[at]}):
    im = cv2.imread(str(ROOT/"data/sppg"/fn)); H, W = im.shape[:2]
    r = pose.predict(im, imgsz=640, conf=0.40, verbose=False)[0]
    if not len(r.boxes): continue
    B = r.boxes.xyxy.cpu().numpy(); K = r.keypoints.data.cpu().numpy()
    ppl = build_rois(B, K); pboxes, verd = dep.perceive(im, B, K)
    crops, idx = [], []
    for i, pp in enumerate(ppl):
        x1, y1, x2, y2 = pp.person_box; pw, ph = x2-x1, y2-y1
        c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
        if c.size and min(c.shape[:2]) >= 16: crops.append(lb(c)); idx.append(i)
    out = {}
    if crops:
        x = torch.from_numpy(np.stack(crops)).cuda()
        with torch.no_grad():
            for name, (kind, net, cuts) in models.items(): out[name] = torch.sigmoid(net(x)).cpu().numpy()
    for at, k in zip(ITEMS, range(3)):
        for pid, tv in lab[at].items():
            if persons[pid]["frame"] != fn: continue
            v = [iou(p.person_box, persons[pid]["person_box"]) for p in ppl]
            if not v: continue
            bi = int(np.argmax(v))
            if v[bi] < 0.3: continue
            row = idx.index(bi) if bi in idx else None
            for name, (kind, net, cuts) in models.items():
                if row is None: alert = 0
                elif kind == "v1": alert = int(out[name][row, k] < cuts[at]["present_below"])
                else: alert = int(out[name][row, 3+k] >= cuts[at]["known_above"] and out[name][row, k] < cuts[at]["present_below"])
                res[name][at].append((tv, alert))
            st = verd[bi]["apd"].get(GRP[at], "unknown") if bi < len(verd) else "unknown"
            res["DEPLOYED pipeline"][at].append((tv, int(st == "violation")))

print("TEST = COCO human labels, pose yolo11s @0.40. Violation positive; abstain or missed = no alert.\n")
for at in ITEMS:
    print(f"{at}:")
    print(f"  {'model':24} {'judgeable':>9} {'alerts':>6} {'prec':>6} {'recall':>7} {'F1':>6} {'MCC':>7} | "
          f"{'unjudgeable':>11} {'flagged':>8}")
    for name in list(models) + ["DEPLOYED pipeline"]:
        rows = res[name][at]
        jud = [(t, q) for t, q in rows if t in ("absent", "present")]
        unc = [q for t, q in rows if t == "unclear"]
        Y = np.array([t == "absent" for t, _ in jud]); Q = np.array([q for _, q in jud])
        print(f"  {name:24} {len(jud):9d} {Q.sum():6d} {precision_score(Y,Q,zero_division=0):6.1%} "
              f"{recall_score(Y,Q,zero_division=0):7.1%} {f1_score(Y,Q,zero_division=0):6.3f} "
              f"{matthews_corrcoef(Y,Q) if len(set(Q))>1 else 0:7.3f} | {len(unc):11d} "
              f"{(np.mean(unc) if unc else 0):8.0%}")
    print()
