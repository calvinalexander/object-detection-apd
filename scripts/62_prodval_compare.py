"""Rank the classifier generations on the production annotation.

The site pool cannot support this comparison: every training set drew from the
same 60 sites, so excluding them leaves 30 / 6 / 6 crops (scripts/61). The
production val is the only set no classifier has trained on, and it is annotated
independently with explicit `no_*` negatives.

One caveat that cannot be measured away: these labels follow the ORIGINAL
head_cap contract, under which a peaked cap counts as compliant. Our rule was
revised on 2026-09-14 to call street hats violations, so on any frame containing
a hat the newer models are penalised for obeying the current rule. head_cap
numbers here are therefore a LOWER bound for hat-sanitised models.
"""
import argparse, collections, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from ultralytics import YOLO
from apd.pose import build_rois, roi_usable
from apd.schema import GEOMETRY, assert_contract

assert_contract()
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ATT = ["head_cap", "mask", "gloves"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dump/dataset/apd_v3/dataset")
ap.add_argument("--gate", type=int, default=1)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", default="artifacts/prodval_compare.json")
a = ap.parse_args()

MODELS = {
    "ci3 (yolo26n-cls)":     ("yolo", "runs/classify/runs/classify/runs/ci3_%s/weights/best.pt"),
    "inpaint (yolo11n-cls)": ("yolo", "runs/classify/runs/classify_inpaint/%s/weights/best.pt"),
    "inpaint (yolo11s-cls)": ("yolo", "runs/classify/runs/classify_sm_experiment/yolo11s-cls_%s/weights/best.pt"),
    "inpaint (yolo11m-cls)": ("yolo", "runs/classify/runs/classify_sm_experiment/yolo11m-cls_%s/weights/best.pt"),
    "inpaint (effnet-b1)":   ("effnet", "runs/classify_effnet_b1/%s_best.pt"),
    "voter4 (original)":     ("yolo", "runs/classify/runs/voter4_%s/weights/best.pt"),
}
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)

def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

def centre_in(b,p):
    cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

# ---- harvest ROIs + ground truth once, so every model sees identical crops
root = Path(a.data)
imgs = sorted(p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
pose = YOLO(".work/yolo11x-pose.pt")
bank = {at: {"crops": [], "y": []} for at in ATT}
gbank = {"pairs": [], "y": []}
for n, p in enumerate(imgs, 1):
    im = cv2.imread(str(p))
    if im is None: continue
    H, W = im.shape[:2]
    gtb = []
    lp = root/"labels/val"/(p.stem+".txt")
    if lp.exists():
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) < 5: continue
            c = int(f[0]); cx,cy,bw,bh = (float(v) for v in f[1:5])
            gtb.append(([(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H], NAMES[c]))
    rr = pose.predict(im, imgsz=1280, conf=0.40, verbose=False, device=0)[0]
    if rr.keypoints is None or not len(rr.boxes): continue
    people = build_rois(rr.boxes.xyxy.cpu().numpy(), rr.keypoints.data.cpu().numpy())
    pb = [q.person_box for q in people]
    for pp in people:
        for at in ATT:
            hp = any(nm==at and centre_in(bb, pp.person_box) for bb,nm in gtb)
            hn = any(nm==f"no_{at}" and centre_in(bb, pp.person_box) for bb,nm in gtb)
            if hp == hn: continue
            y = 1 if hp else 0
            if at == "gloves":
                hs = []
                for h in pp.hands[:2]:
                    if a.gate and not roi_usable(h, im.shape, part="hand", others=pb, self_box=pp.person_box):
                        continue
                    c_ = crop_np(im, h, GEOMETRY.hand_crop_px)
                    if c_ is not None: hs.append(c_)
                if len(hs) < 2: continue
                gbank["pairs"].append(hs[:2]); gbank["y"].append(y)
            else:
                if a.gate and not roi_usable(pp.head, im.shape, part="head", others=pb, self_box=pp.person_box):
                    continue
                hc = crop_np(im, pp.head, GEOMETRY.head_crop_px) if pp.head is not None else None
                if hc is None: continue
                bank[at]["crops"].append(hc); bank[at]["y"].append(y)
    if n % 40 == 0: print(f"  harvest {n}/{len(imgs)}", flush=True)

def load_effnet(path):
    net = torchvision.models.efficientnet_b1(weights=None)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    n_out = sd["classifier.1.weight"].shape[0]
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, n_out)
    net.load_state_dict(sd); return net.cuda().eval(), n_out

def probs(kind, path, crops):
    if kind == "yolo":
        m = YOLO(path); idx = [k for k,v in m.names.items() if v == "present"][0]
        out = []
        for i in range(0, len(crops), 64):
            for r in m.predict(crops[i:i+64], imgsz=128, verbose=False, device=0):
                out.append(float(r.probs.data[idx]))
        return np.array(out)
    net, n_out = load_effnet(path); out = []
    for i in range(0, len(crops), 64):
        b = [((cv2.cvtColor(cv2.resize(c,(128,128)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1)
             for c in crops[i:i+64]]
        with torch.no_grad():
            lo = net(torch.tensor(np.array(b)).cuda())
            pr = torch.softmax(lo,1)[:,1] if n_out == 2 else torch.sigmoid(lo[:,0])
        out.append(pr.cpu().numpy())
    return np.concatenate(out)

def boot(y, s, n):
    rng = np.random.default_rng(0); v = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k])) > 1: v.append(roc_auc_score(y[k], s[k]))
    return (float(np.percentile(v,2.5)), float(np.percentile(v,97.5))) if v else (float("nan"),)*2

report = {}
for at in ATT:
    if at == "gloves":
        y = np.array(gbank["y"]); flat = [c for pair in gbank["pairs"] for c in pair]
    else:
        y = np.array(bank[at]["y"]); flat = bank[at]["crops"]
    if len(y) < 12 or len(set(y)) < 2:
        print(f"\n{at}: n={len(y)} — unmeasurable"); report[at] = {"status": "unmeasurable", "n": int(len(y))}; continue
    print(f"\n{at}: n={len(y)} ({int(y.sum())} present / {int((1-y).sum())} absent), gate={'on' if a.gate else 'off'}")
    print(f"  {'model':26} {'AUC':>6} {'acc':>6} | {'VIOLATION (absent) is positive':^30} | {'compliant':^16}")
    print(f"  {'':26} {'':6} {'':6} | {'prec':>7} {'rec':>7} {'F1':>7} {'n':>5} | {'prec':>7} {'F1':>7}")
    report[at] = {"n": int(len(y)), "present": int(y.sum()), "models": {}}
    for name,(kind,tmpl) in MODELS.items():
        path = tmpl % at
        if not Path(path).exists(): continue
        try:
            s = probs(kind, path, flat)
            if at == "gloves":                       # both hands required -> weaker hand
                s = np.minimum(s[0::2], s[1::2])
        except Exception as e:
            print(f"  {name:26}  failed: {type(e).__name__}: {e}"); continue
        auc = roc_auc_score(y, s); lo, hi = boot(y, s, a.boot)
        acc = ((s >= .5).astype(int) == y).mean()
        # The alert fires on a violation, so `absent` is the positive class:
        # precision here is "of the violations we raise, how many are real".
        pred = (s >= .5).astype(int)                 # 1 = present/compliant
        yv, pv = 1 - y, 1 - pred                     # flip: 1 = violation
        pr, rc, f1, sup = precision_recall_fscore_support(yv, pv, average="binary", zero_division=0)
        pr2, _, f12, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        report[at]["models"][name] = {"auc": float(auc), "ci":[lo,hi], "acc": float(acc),
                                      "violation": {"precision": float(pr), "recall": float(rc),
                                                    "f1": float(f1), "support": int(yv.sum())},
                                      "compliant": {"precision": float(pr2), "f1": float(f12)}}
        print(f"  {name:26} {auc:6.3f} {acc:6.1%} | {pr:7.1%} {rc:7.1%} {f1:7.3f} {int(yv.sum()):5d} | {pr2:7.1%} {f12:7.3f}")

Path(a.out).write_text(json.dumps(report, indent=2))
print(f"\nwrote {a.out}")
