"""Raise precision by moving the operating point, calibrated off the eval set.

A violation is flagged when p(present) falls below a threshold. At 0.5 the
models sit at 74-78% precision on gloves; a lower threshold flags only the
confident cases. The threshold is chosen on the INTERNAL validation split
(site-grouped, carved from training) to hit a target precision, then applied
unchanged to the human-labelled frames. Choosing it on the eval set would be
fitting to the test and would report a precision that does not exist.

Also tested: requiring several models to agree before flagging, which is the
same idea the teacher used -- disagreement is evidence the evidence is thin.
"""
import argparse, collections, glob, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from sklearn.metrics import precision_recall_fscore_support
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd.schema import GEOMETRY, assert_contract

assert_contract()
ATTRS = ["head_cap", "mask", "gloves"]
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
dev = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="0.85,0.90,0.95")
ap.add_argument("--out", default="artifacts/precision_tuning.json")
a = ap.parse_args()

def norm(im):
    x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
    return x.transpose(2,0,1)
def torch_model(arch, path):
    net = getattr(torchvision.models, arch)(weights=None)
    if hasattr(net, "classifier") and isinstance(net.classifier, torch.nn.Sequential):
        net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
    elif hasattr(net, "fc"):
        net.fc = torch.nn.Linear(net.fc.in_features, 1)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return net.to(dev).eval()

CAND = {
  "resnet34": ("torch","resnet34","runs/bakeoff_torch/resnet34/%s_best.pt"),
  "effnet-b0": ("torch","efficientnet_b0","runs/bakeoff_torch/efficientnet_b0/%s_best.pt"),
  "yolo26n-cls": ("yolo",None,"runs/classify/runs/bakeoff/yolo26n-cls_%s/weights/best.pt"),
  "yolo11s-cls": ("yolo",None,"runs/classify/runs/bakeoff/yolo11s-cls_%s/weights/best.pt"),
}
models = {}
for n,(k,arch,t) in CAND.items():
    per = {at: (torch_model(arch, t % at) if k=="torch" else YOLO(t % at))
           for at in ATTRS if Path(t % at).exists()}
    if per: models[n] = (k, per)

def probs(name, at, crops):
    k, per = models[name]
    if at not in per or not crops: return np.zeros(0)
    if k == "torch":
        outs = []
        for i in range(0, len(crops), 64):
            with torch.no_grad():
                x = torch.tensor(np.array([norm(cv2.resize(c,(128,128))) for c in crops[i:i+64]])).to(dev)
                outs.append(torch.sigmoid(per[at](x)[:,0]).cpu().numpy())
        return np.concatenate(outs)
    m = per[at]; idx = [kk for kk,v in m.names.items() if v=="present"][0]
    outs = []
    for i in range(0, len(crops), 64):
        outs += [float(r.probs.data[idx]) for r in m.predict(crops[i:i+64], imgsz=128, verbose=False, device=0)]
    return np.array(outs)

# ---------- calibration set: the internal val split ----------
print("calibrating on the internal validation split (never the eval frames)")
cal = {}
for at in ATTRS:
    items = [(p, 1 if cls=="present" else 0) for cls in ("present","absent")
             for p in glob.glob(f"data/crops_cocoheld/{at}/val/{cls}/*.jpg")]
    crops = [cv2.imread(p) for p, _ in items]
    y = np.array([1-yy for _, yy in items])          # 1 = violation
    cal[at] = {"y": y, "p": {n: probs(n, at, crops) for n in models}}
    print(f"  {at}: {len(y)} crops, {int(y.sum())} violations")

def pick(y, p_present, target):
    """Lowest threshold on p(present) that reaches the target violation precision."""
    best = None
    for t in np.arange(0.50, 0.01, -0.01):
        pred = (p_present < t).astype(int)
        if pred.sum() == 0: continue
        pr = (pred & y).sum() / pred.sum()
        rc = (pred & y).sum() / max(1, y.sum())
        if pr >= target and (best is None or rc > best[2]):
            best = (float(t), float(pr), float(rc))
    return best

# ---------- evaluation on human labels ----------
persons = {json.loads(l)["person_id"]: json.loads(l) for l in Path("artifacts/persons.jsonl").read_text().splitlines()}
truth = collections.defaultdict(dict)
for line in Path("artifacts/seed_labels.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["person_id"] in persons:
        for at,v in (r.get("labels") or {}).items():
            if v in ("present","absent"): truth[at][r["person_id"]] = v
def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

ev = {at: {"y": [], "p": collections.defaultdict(list)} for at in ATTRS}
frames = sorted({persons[p]["frame"] for at in ATTRS for p in truth[at]})
for k, fname in enumerate(frames, 1):
    im = cv2.imread(str(Path("data/sppg")/fname))
    if im is None: continue
    for at in ATTRS:
        ids = [pid for pid in truth[at] if persons[pid]["frame"] == fname]
        if not ids: continue
        for pid in ids:
            rec = persons[pid]
            ev[at]["y"].append(0 if truth[at][pid] == "present" else 1)
            if at == "gloves":
                hs = [c for c in (crop_np(im,h,GEOMETRY.hand_crop_px) for h in (rec["hand_rois"] or [])[:2]) if c is not None]
                for n in models:
                    ev[at]["p"][n].append(1.0 if len(hs) < 2 else float(min(probs(n, at, hs[:2]))))
            else:
                hc = crop_np(im, rec["head_roi"], GEOMETRY.head_crop_px) if rec.get("head_roi") else None
                for n in models:
                    ev[at]["p"][n].append(1.0 if hc is None else float(probs(n, at, [hc])[0]))
    if k % 20 == 0: print(f"  eval {k}/{len(frames)}", flush=True)

PROD = {"head_cap": (.658,.269,.382), "mask": (.933,.368,.528), "gloves": (.818,.098,.175)}
targets = [float(x) for x in a.targets.split(",")]
out = {}
for at in ATTRS:
    y = np.array(ev[at]["y"])
    print(f"\n=== {at}: {len(y)} persons, {int(y.sum())} violations ===")
    pp, pr_, rc_ = "production", *PROD[at][:2]
    print(f"  {'model @ operating point':34} {'thr':>5} {'prec':>7} {'recall':>7} {'F1':>7}")
    print(f"  {'apd_mbg (production)':34} {'-':>5} {PROD[at][0]:7.1%} {PROD[at][1]:7.1%} {PROD[at][2]:7.3f}")
    out[at] = {}
    for n in models:
        pe = np.array(ev[at]["p"][n])
        for tgt in targets:
            got = pick(cal[at]["y"], cal[at]["p"][n], tgt)
            if got is None: continue
            t, cpr, crc = got
            pred = (pe < t).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            out[at][f"{n}@{tgt}"] = {"thr": t, "cal_prec": cpr, "precision": pr, "recall": rc, "f1": f1}
            print(f"  {n+' (target '+str(int(tgt*100))+'%)':34} {t:5.2f} {pr:7.1%} {rc:7.1%} {f1:7.3f}")
    # agreement of all four models, at their 0.90-target thresholds
    ths = {n: (pick(cal[at]["y"], cal[at]["p"][n], 0.90) or (0.5,0,0))[0] for n in models}
    votes = np.array([(np.array(ev[at]["p"][n]) < ths[n]).astype(int) for n in models])
    for need, label in ((len(models), "unanimous"), (max(2, len(models)-1), "majority")):
        pred = (votes.sum(0) >= need).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        out[at][f"agree/{label}"] = {"precision": pr, "recall": rc, "f1": f1}
        print(f"  {'ensemble '+label+' (@90% targets)':34} {'-':>5} {pr:7.1%} {rc:7.1%} {f1:7.3f}")
Path(a.out).write_text(json.dumps(out, indent=2))
print(f"\nwrote {a.out}")
