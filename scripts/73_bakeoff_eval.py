"""Score every variant on the human labels and test the gap to production.

"Better" is not established by a larger F1 on 44-202 people. The comparison is
paired -- every model judges the same persons -- so McNemar's test applies: of
the cases where two models disagree, how lopsided is the disagreement. A model
is reported as beating production only when that asymmetry is unlikely by chance.

Operational scoring throughout: an abstention counts as no alert raised.
"""
import collections, glob, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision, openvino as ov
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd.schema import GEOMETRY, assert_contract

assert_contract()
ATTRS = ["head_cap", "mask", "gloves"]
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
dev = "cuda"

def norm(im):
    x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
    return x.transpose(2,0,1)

class Shared(torch.nn.Module):
    """A trunk with N heads -- the attribute selects an output column."""
    def __init__(self, arch, nout=3):
        super().__init__()
        b = getattr(torchvision.models, arch)(weights=None)
        if hasattr(b, "classifier") and isinstance(b.classifier, torch.nn.Sequential):
            d = b.classifier[-1].in_features; b.classifier = torch.nn.Identity()
        elif hasattr(b, "fc"):
            d = b.fc.in_features; b.fc = torch.nn.Identity()
        self.trunk, self.head = b, torch.nn.Linear(d, nout)
    def forward(self, x): return self.head(self.trunk(x))


def shared_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Shared(ck["arch"]); m.load_state_dict(ck["model"])
    return m.cuda().eval(), ck["attrs"]


def torch_model(arch, path):
    net = getattr(torchvision.models, arch)(weights=None)
    if hasattr(net, "classifier") and isinstance(net.classifier, torch.nn.Sequential):
        net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
    elif hasattr(net, "fc"):
        net.fc = torch.nn.Linear(net.fc.in_features, 1)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return net.to(dev).eval()

CANDIDATES = {
    "effnet-b1 (published)": ("torch", "efficientnet_b1", "runs/effnet_coco_holdout/%s_last.pt"),
    "effnet-b0 (regularised)": ("torch", "efficientnet_b0", "runs/bakeoff_torch/efficientnet_b0/%s_best.pt"),
    "resnet34 (regularised)": ("torch", "resnet34", "runs/bakeoff_torch/resnet34/%s_best.pt"),
    "yolo26n-cls": ("yolo", None, "runs/classify/runs/bakeoff/yolo26n-cls_%s/weights/best.pt"),
    "yolo11s-cls": ("yolo", None, "runs/classify/runs/bakeoff/yolo11s-cls_%s/weights/best.pt"),
    "yolo26s-cls": ("yolo", None, "runs/classify/runs/bakeoff/yolo26s-cls_%s/weights/best.pt"),
    "effnet-b1 (regularised)": ("torch", "efficientnet_b1", "runs/bakeoff_torch/efficientnet_b1/%s_best.pt"),
    "effnet-b0 +SDK data": ("torch", "efficientnet_b0", "runs/bakeoff_sdk/efficientnet_b0/%s_best.pt"),
    "resnet34 +SDK data": ("torch", "resnet34", "runs/bakeoff_sdk/resnet34/%s_best.pt"),
    "resnet34 LABELS-FIXED": ("torch", "resnet34", "runs/fixed_base/resnet34/%s_best.pt"),
    "effnet-b0 +SDK LABELS-FIXED": ("torch", "efficientnet_b0", "runs/fixed_sdk/efficientnet_b0/%s_best.pt"),
    "resnet34 +SDK LABELS-FIXED": ("torch", "resnet34", "runs/fixed_sdk/resnet34/%s_best.pt"),
}
TWOTRUNK = {
    "2TRUNK resnet34": "runs/twotrunk/resnet34",
    "2TRUNK effnet-b0": "runs/twotrunk/efficientnet_b0",
}
SHARED = {
    "SHARED effnet-b0": "runs/shared/efficientnet_b0/shared_best.pt",
    "SHARED resnet34":  "runs/shared/resnet34/shared_best.pt",
}
models = {}
for name, d in TWOTRUNK.items():
    per = {}
    for tag in ("head", "hand"):
        p = Path(d) / f"{tag}_best.pt"
        if not p.exists(): continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        net = Shared(ck["arch"], len(ck["attrs"])); net.load_state_dict(ck["model"])
        net = net.cuda().eval()
        for i, at in enumerate(ck["attrs"]):
            per[at] = (net, i)
    if per: models[name] = ("shared", per)
for name, path in SHARED.items():
    if Path(path).exists():
        net, attrs = shared_model(path)
        models[name] = ("shared", {at: (net, attrs.index(at)) for at in attrs})
for name, (kind, arch, tmpl) in CANDIDATES.items():
    per = {}
    for at in ATTRS:
        p = tmpl % at
        if not Path(p).exists(): continue
        per[at] = torch_model(arch, p) if kind == "torch" else YOLO(p)
    if per: models[name] = (kind, per)
print("candidates:", ", ".join(models))

def probs(name, at, crops):
    kind, per = models[name]
    if at not in per: return None
    if kind == "shared":
        net, col = per[at]
        with torch.no_grad():
            x = torch.tensor(np.array([norm(cv2.resize(c,(128,128))) for c in crops])).to(dev)
            return torch.sigmoid(net(x)[:, col]).cpu().numpy()
    if kind == "torch":
        with torch.no_grad():
            x = torch.tensor(np.array([norm(cv2.resize(c,(128,128))) for c in crops])).to(dev)
            return torch.sigmoid(per[at](x)[:,0]).cpu().numpy()
    m = per[at]; idx = [k for k,v in m.names.items() if v=="present"][0]
    return np.array([float(r.probs.data[idx]) for r in m.predict(crops, imgsz=128, verbose=False, device=0)])

core = ov.Core(); dv = "GPU" if "GPU" in core.available_devices else "CPU"
PROD = {
    "apd_mbg Dec-2025 (prod)": "model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml",
    "apd_mbg Mar-2026 (prod)": "artifacts/sdk/sdk/build/apd_mbg_openvino_model/best.xml",
}
mbgs = {k: core.compile_model(core.read_model(v), dv) for k, v in PROD.items() if Path(v).exists()}
mbg = list(mbgs.values())[0]
port = mbg.outputs[0]
def letterbox(img, s=640):
    h,w = img.shape[:2]; r = min(s/h, s/w); nh,nw = round(h*r), round(w*r)
    o = np.full((s,s,3),114,np.uint8); t,l = (s-nh)//2,(s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img,(nw,nh)); return o,r,l,t
def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))
def centre_in(b,p):
    cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

persons = {json.loads(l)["person_id"]: json.loads(l) for l in Path("artifacts/persons.jsonl").read_text().splitlines()}
truth = collections.defaultdict(dict)
for line in Path("artifacts/seed_labels.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["person_id"] not in persons: continue
    for at, v in (r.get("labels") or {}).items():
        if v in ("present","absent"): truth[at][r["person_id"]] = v

preds = {at: collections.defaultdict(list) for at in ATTRS}   # attr -> model -> [pred]
ys = {at: [] for at in ATTRS}
frames = sorted({persons[p]["frame"] for at in ATTRS for p in truth[at]})
for k, fname in enumerate(frames, 1):
    im = cv2.imread(str(Path("data/sppg")/fname))
    if im is None: continue
    img, r, dx, dy = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ppe_by = {}
    for pname, pnet in mbgs.items():
      o = pnet([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[pnet.outputs[0]][0]
      if o.shape[0] < o.shape[1]: o = o.T
      sc = o[:,4:11]; cl = sc.argmax(1); cf = sc.max(1); m = cf >= 0.30
      b, cf, cl = o[m][:,:4], cf[m], cl[m]; ppe = []
      if len(b):
        xy = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2],1)
        xy[:,[0,2]] -= dx; xy[:,[1,3]] -= dy; xy /= r
        idx = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                               cf.tolist(), 0.30, 0.7)
        idx = np.array(idx).flatten() if len(idx) else np.zeros(0,int)
        ppe = [(xy[i], NAMES[cl[i]]) for i in idx if NAMES[cl[i]] != "person"]
      ppe_by[pname] = ppe
    for at in ATTRS:
        for pid, tv in truth[at].items():
            rec = persons[pid]
            if rec["frame"] != fname: continue
            pbox = rec["person_box"]
            ys[at].append(0 if tv == "present" else 1)
            mp = any(nm==at and centre_in(bb, pbox) for bb,nm in ppe)
            mn = any(nm==f"no_{at}" and centre_in(bb, pbox) for bb,nm in ppe)
            for pname, pnet in mbgs.items():
                pp_ = ppe_by[pname]
                a_ = any(nm==at and centre_in(bb, pbox) for bb,nm in pp_)
                b_ = any(nm==f"no_{at}" and centre_in(bb, pbox) for bb,nm in pp_)
                preds[at][pname].append(0 if a_ == b_ else (0 if a_ else 1))
            if at == "gloves":
                hs = [c for c in (crop_np(im,h,GEOMETRY.hand_crop_px) for h in (rec["hand_rois"] or [])[:2]) if c is not None]
                for name in models:
                    if at not in models[name][1]: continue
                    if len(hs) < 2: preds[at][name].append(0); continue
                    pr = probs(name, at, hs[:2])
                    preds[at][name].append(0 if pr is None else (0 if min(pr) >= .5 else 1))
            else:
                hc = crop_np(im, rec["head_roi"], GEOMETRY.head_crop_px) if rec.get("head_roi") else None
                for name in models:
                    if at not in models[name][1]: continue
                    if hc is None: preds[at][name].append(0); continue
                    pr = probs(name, at, [hc])
                    preds[at][name].append(0 if pr is None else (0 if pr[0] >= .5 else 1))
    if k % 15 == 0: print(f"  {k}/{len(frames)}", flush=True)

print("\n=== operational scores on human labels, and McNemar vs production ===")
summary = {}
for at in ATTRS:
    y = np.array(ys[at]); base = np.array(preds[at]["apd_mbg Dec-2025 (prod)"])
    print(f"\n{at}: {len(y)} persons, {int(y.sum())} violations")
    bp = y.mean()
    print(f"  {'model':26} {'prec':>7} {'recall':>7} {'F1':>7} {'MCC':>6} {'bal-acc':>8} | {'p vs prod':>10}")
    print(f"  {'ALWAYS-FLAG baseline':26} {bp:7.1%} {1.0:7.1%} {2*bp/(1+bp):7.3f} {0.0:6.3f} {0.5:8.1%} | {'-':>10}")
    bc = (base == y)
    for name in list(mbgs) + [m for m in models if len(preds[at].get(m, [])) == len(y)]:
        p = np.array(preds[at][name])
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
        mcc = matthews_corrcoef(y, p) if len(set(p)) > 1 else 0.0
        ba = balanced_accuracy_score(y, p)
        if name.startswith("apd_mbg"):
            print(f"  {name:26} {pr:7.1%} {rc:7.1%} {f1:7.3f} {mcc:6.3f} {ba:8.1%} | {'baseline':>10}")
            continue
        mc = (p == y)
        b_ = int(np.sum(mc & ~bc)); w_ = int(np.sum(~mc & bc))
        pv = binomtest(b_, b_+w_, 0.5).pvalue if (b_+w_) else 1.0
        star = "*" if pv < 0.05 and b_ > w_ else " "
        print(f"  {name:26} {pr:7.1%} {rc:7.1%} {f1:7.3f} {mcc:6.3f} {ba:8.1%} | {pv:9.2e}{star}")
        summary[f"{at}/{name}"] = {"f1": float(f1), "mcc": float(mcc), "bal_acc": float(ba),
                                   "precision": float(pr), "recall": float(rc),
                                   "better": b_, "worse": w_, "p": float(pv)}
Path("artifacts/bakeoff_eval.json").write_text(json.dumps(summary, indent=2))
print("\nbetter/worse = persons this model gets right and production gets wrong, and vice versa")
print("p = two-sided binomial (McNemar) on those disagreements; ** = significant at 0.05")
