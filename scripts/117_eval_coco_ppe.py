"""TEST: every retrained PPE model against the deployed one on the COCO human labels, end to end.

The COCO-annotated frames (data/sppg, labels in artifacts/seed_labels.jsonl) were
never used for training or selection. Each model gets exactly what the device
gives it:

    yolo11s-pose @ conf 0.40 (the bundle's pose model)  ->  build_rois
    head_cap : head ROI, device head gate (roi_usable, no collision)  -> _crop 128
    mask     : same head gate + facing gate (nose >= 0.5, not turned away) -> face_roi 0.8 -> _crop 128
    gloves   : crop detector on the pose person box (+18% pad), conf 0.25, higher of gloves/no_gloves
    PERSON   : one-network models (scripts/118) on that same person crop, all three items at once,
               judging every located person (no head/facing gate: that is the design)

Scoring is operational, as in every earlier comparison: violation (item absent)
is the positive class, an abstention or an unmatched person is "no alert".
Thresholds are the ones each model picked on VAL. McNemar is against the
deployed model on the same people.

    python scripts/117_eval_coco_ppe.py [--pose yolo11s-pose.pt]
Writes runs/retrain_clean/coco_test.json and prints the tables.
"""
import argparse, collections, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from scipy.stats import chi2
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd.pose import build_rois, face_roi, roi_usable
from apd.schema import GEOMETRY
from apd_perception_ours import _crop, _prep1, _facing_camera
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--pose", default="yolo11s-pose.pt")
ap.add_argument("--conf", type=float, default=0.40)
ap.add_argument("--runs", default="runs/retrain_clean")
a = ap.parse_args()
R = ROOT/a.runs
NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
PROD = {"head_cap": 0.191, "mask": 0.146, "gloves": 0.047}     # apd_mbg Dec-2025, same people (section 6d)

# ------------------------------------------------------------------ models
def torch_net(arch, w):
    net = getattr(torchvision.models, arch)(weights=None)
    if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, 1)
    else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
    sd = torch.load(w, map_location="cpu", weights_only=False)
    net.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd)
    return net.eval().cuda()


class Clf:
    def __init__(self, spec):
        self.spec = spec; self.t = spec["threshold"]
        if spec["kind"] == "torch": self.net = torch_net(spec["arch"], spec["weights"])
        else: self.net = YOLO(spec["weights"]); self.pi = spec["present_index"]

    @torch.no_grad()
    def p(self, crops):
        if not crops: return []
        if self.spec["kind"] == "torch":
            x = torch.from_numpy(np.stack([_prep1(c) for c in crops])).cuda()
            return torch.sigmoid(self.net(x)[:, 0]).cpu().tolist()
        return [float(r.probs.data[self.pi]) for r in self.net.predict(list(crops), imgsz=128, verbose=False)]


clf = {"head_cap": {"DEPLOYED resnet34 (fixed_sdk)": {"kind": "torch", "arch": "resnet34", "threshold": 0.5,
                     "weights": str(ROOT/"runs/fixed_sdk/resnet34/head_cap_best.pt")}},
       "mask": {"DEPLOYED effnet_b0 face (g0.8)": {"kind": "torch", "arch": "efficientnet_b0", "threshold": 0.30,
                 "weights": str(ROOT/"runs/mask_face/g0.8_deployed/mask_best.pt")}}}
for at in clf:
    s = R/at/"summary.json"
    if s.exists():
        for k, v in json.loads(s.read_text()).items(): clf[at][k] = v
clf = {at: {k: Clf(v) for k, v in d.items()} for at, d in clf.items()}
dets = {"DEPLOYED yolo26n (det7)": ROOT/"runs/det7/yolo26n/weights/best.pt"}
for t in ("yolo26n", "yolo26s", "yolo26m"):
    w = R/"gloves_det"/t/"weights/best.pt"
    if w.exists(): dets[t + " (det7 clean)"] = w
dets = {k: YOLO(str(v)) for k, v in dets.items()}

# one-network person models (scripts/118): person crop -> [head_cap, mask, gloves]
PMEAN = np.array([0.485, 0.456, 0.406], np.float32); PSTD = np.array([0.229, 0.224, 0.225], np.float32)
def letterbox(im, s):
    h, w = im.shape[:2]; r = s / max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    out = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    out[y:y+im.shape[0], x:x+im.shape[1]] = im
    return out
pmodels = {}
for sp in sorted((ROOT/"runs/person_ppe").glob("*/summary.json")) if (ROOT/"runs/person_ppe").exists() else []:
    j = json.loads(sp.read_text())
    net = getattr(torchvision.models, j["arch"])(weights=None)
    if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, 3)
    else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 3)
    net.load_state_dict(torch.load(j["weights"], map_location="cpu", weights_only=False))
    pmodels[f"PERSON {j['arch']}_{j['px']}" + (f" +{j['tag']}" if j.get('tag') else "")] = (net.eval().cuda(), j)


@torch.no_grad()
def person_probs(net, px, crops):
    x = np.stack([((cv2.cvtColor(letterbox(c, px), cv2.COLOR_BGR2RGB).astype(np.float32)/255 - PMEAN)/PSTD).transpose(2, 0, 1) for c in crops])
    return torch.sigmoid(net(torch.from_numpy(x).cuda())).cpu().numpy()

# ------------------------------------------------------------------ COCO truth
persons = {json.loads(l)["person_id"]: json.loads(l) for l in open(ROOT/"artifacts/persons.jsonl")}
truth = collections.defaultdict(dict)
for l in open(ROOT/"artifacts/seed_labels.jsonl"):
    r = json.loads(l)
    if r["person_id"] in persons:
        for at, v in (r.get("labels") or {}).items():
            if v in ("present", "absent"): truth[at][r["person_id"]] = v
frames = sorted({persons[p]["frame"] for at in truth for p in truth[at]})
pose = YOLO(str(ROOT/a.pose))


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix*iy
    return i / max(1e-6, (p[2]-p[0])*(p[3]-p[1]) + (q[2]-q[0])*(q[3]-q[1]) - i)


Y = collections.defaultdict(list)
Q = {at: collections.defaultdict(list) for at in ("head_cap", "mask", "gloves")}
located = collections.Counter()
for fn in frames:
    im = cv2.imread(str(ROOT/"data/sppg"/fn))
    if im is None: continue
    H, W = im.shape[:2]
    r = pose.predict(im, imgsz=640, conf=a.conf, verbose=False)[0]
    B = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
    K = r.keypoints.data.cpu().numpy() if len(r.boxes) else np.zeros((0, 17, 3))
    ppl = build_rois(B, K) if len(B) else []
    pboxes = [p.person_box for p in ppl]
    heads, faces, gcrops = {}, {}, {}
    for i, pp in enumerate(ppl):
        ok = (pp.head is not None and not pp.head_collision and
              roi_usable(pp.head, im.shape, part="head", others=pboxes, self_box=pp.person_box))
        if ok:
            c = _crop(im, pp.head, GEOMETRY.head_crop_px)
            if c is not None: heads[i] = c
            if _facing_camera(K[i], 0.5, True):
                fb = face_roi(K[i], tuple(B[i]), gain=0.8)
                fc = None if fb is None else _crop(im, fb, GEOMETRY.face_crop_px)
                if fc is not None: faces[i] = fc
        x1, y1, x2, y2 = pp.person_box; pw, ph = x2-x1, y2-y1
        g = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
        if g.size and min(g.shape[:2]) >= 16: gcrops[i] = g
    verdict = {"head_cap": {}, "mask": {}, "gloves": {}}          # model -> {person index: 1 violation / 0}
    for at, crops in (("head_cap", heads), ("mask", faces)):
        idx = list(crops)
        for name, m in clf[at].items():
            ps = m.p([crops[i] for i in idx])
            verdict[at][name] = {i: int(p < m.t) for i, p in zip(idx, ps)}
    for name, m in dets.items():
        v = {}
        for i, g in gcrops.items():
            rr = m.predict(g, imgsz=416, conf=0.25, verbose=False)[0]
            best = {}
            for c, s in zip(rr.boxes.cls.cpu().numpy(), rr.boxes.conf.cpu().numpy()):
                n = NAMES[int(c)]
                if n in ("gloves", "no_gloves"): best[n] = max(best.get(n, 0), float(s))
            if best: v[i] = int(best.get("no_gloves", 0) > best.get("gloves", 0))
        verdict["gloves"][name] = v
    idx = list(gcrops)
    for name, (net, j) in pmodels.items():
        P_ = person_probs(net, j["px"], [gcrops[i] for i in idx]) if idx else np.zeros((0, 3))
        for k, at in enumerate(("head_cap", "mask", "gloves")):
            t = j["items"][at]["threshold"]
            verdict[at][name] = {i: int(P_[n, k] < t) for n, i in enumerate(idx)}
            if at != "gloves":                         # same abstention as the device: head gate (+ facing for mask)
                keep = heads if at == "head_cap" else faces
                verdict[at][name + " +gate"] = {i: int(P_[n, k] < t) for n, i in enumerate(idx) if i in keep}
    for at in Q:
        for pid, tv in truth[at].items():
            if persons[pid]["frame"] != fn: continue
            Y[at].append(1 if tv == "absent" else 0)
            ious = [iou(p, persons[pid]["person_box"]) for p in pboxes]
            bi = int(np.argmax(ious)) if ious else -1
            hit = bi >= 0 and ious[bi] >= 0.3
            located[(at, hit)] += 1
            for name, v in verdict[at].items():
                Q[at][name].append(v.get(bi, 0) if hit else 0)


def mcnemar(y, q0, q1):
    a0, a1 = q0 == y, q1 == y
    b, c = int((a0 & ~a1).sum()), int((~a0 & a1).sum())
    return 1.0 if b + c == 0 else float(chi2.sf((abs(b-c)-1)**2/(b+c), 1))


out = {}
print(f"TEST = COCO human labels, {len(frames)} frames, pose {a.pose} @ {a.conf}\n")
for at in Q:
    y = np.array(Y[at]); loc = located[(at, True)]
    print(f"{at}: {len(y)} labelled people ({int(y.sum())} violations), pose located {loc} ({loc/len(y):.0%})")
    print(f"  {'model':34} {'val MCC':>7} {'thr':>5} {'prec':>6} {'recall':>6} {'F1':>6} {'MCC':>6}  {'p vs deployed':>13}")
    base = next(k for k in Q[at] if k.startswith("DEPLOYED"))
    rows = []
    for name, q in Q[at].items():
        q = np.array(q)
        m = matthews_corrcoef(y, q) if len(set(q.tolist())) > 1 else 0.0
        spec = (clf[at][name].spec if at in clf and name in clf[at] else
                {"val_mcc": pmodels[name.replace(" +gate", "")][1]["items"][at]["val_mcc"],
                 "threshold": pmodels[name.replace(" +gate", "")][1]["items"][at]["threshold"]}
                if name.replace(" +gate", "") in pmodels else {})
        rows.append((name, spec.get("val_mcc"), spec.get("threshold", 0.25), precision_score(y, q, zero_division=0),
                     recall_score(y, q, zero_division=0), f1_score(y, q, zero_division=0), m,
                     "baseline" if name == base else f"{mcnemar(y, np.array(Q[at][base]), q):.3f}"))
    rows.sort(key=lambda r: -r[6])
    for n, vm, t, p, rc, f, m, pv in rows:
        print(f"  {n:34} {('%.3f' % vm) if vm is not None else '   -':>7} {t:5.2f} {p:6.1%} {rc:6.1%} {f:6.3f} {m:6.3f}  {pv:>13}")
    print(f"  {'production apd_mbg (reference)':34} {'':>7} {'':>5} {'':>6} {'':>6} {'':>6} {PROD[at]:6.3f}\n")
    out[at] = [dict(zip(["model", "val_mcc", "threshold", "precision", "recall", "f1", "mcc", "p_vs_deployed"], r)) for r in rows]
(R/"coco_test.json").write_text(json.dumps(out, indent=2))
