"""EfficientNet-B1 (ours) vs apd_mbg (production), same persons, same labels.

Like-for-like is not free here: the two answer differently. apd_mbg is a 7-class
detector that may emit neither a positive nor a negative box for a person, in
which case it says nothing. Our classifier commits on any ROI the gate admits.
Scoring only where each model spoke flatters whichever abstains more, so both
views are reported:

  committed   precision/recall over instances that model actually answered
  operational abstention counted as "no alert" -- a violation nobody flags is a
              violation missed, which is what the deployment actually delivers

Violation (`absent`) is the positive class throughout: the system raises alerts
on violations, so precision here means "of the alerts raised, how many are real".
"""
import argparse, collections, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision, openvino as ov
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO
from apd.pose import build_rois, roi_usable
from apd.schema import GEOMETRY, assert_contract

assert_contract()
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ATT = ["head_cap", "mask", "gloves"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dump/dataset/apd_v3/dataset")
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--gate", type=int, default=1)
ap.add_argument("--out", default="artifacts/effnet_vs_prod.json")
a = ap.parse_args()

MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
def load_effnet(path):
    net = torchvision.models.efficientnet_b1(weights=None)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    n_out = sd["classifier.1.weight"].shape[0]
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, n_out)
    net.load_state_dict(sd); return net.cuda().eval(), n_out
eff = {k: load_effnet(f"runs/classify_effnet_b1/{k}_best.pt") for k in ATT}

def eff_prob(at, crops):
    net, n_out = eff[at]
    b = [((cv2.cvtColor(cv2.resize(c,(128,128)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1)
         for c in crops]
    with torch.no_grad():
        lo = net(torch.tensor(np.array(b)).cuda())
        pr = torch.softmax(lo,1)[:,1] if n_out == 2 else torch.sigmoid(lo[:,0])
    return pr.cpu().numpy()

core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
mbg = core.compile_model(core.read_model("model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml"), dev)
port = mbg.outputs[0]
pose = YOLO(".work/yolo11x-pose.pt")

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

root = Path(a.data)
imgs = sorted(p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
rows = collections.defaultdict(list)      # attr -> (truth_violation, eff_pred, mbg_pred)  pred: 1 viol, 0 ok, -1 abstain
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

    img, r, dx, dy = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    o = mbg([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
    if o.shape[0] < o.shape[1]: o = o.T
    sc = o[:,4:11]; cl = sc.argmax(1); cf = sc.max(1); k = cf >= a.conf
    b, cf, cl = o[k][:,:4], cf[k], cl[k]; ppe = []
    if len(b):
        xy = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2],1)
        xy[:,[0,2]] -= dx; xy[:,[1,3]] -= dy; xy /= r
        idx = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                               cf.tolist(), a.conf, 0.7)
        idx = np.array(idx).flatten() if len(idx) else np.zeros(0,int)
        ppe = [(xy[i], NAMES[cl[i]]) for i in idx if NAMES[cl[i]] != "person"]

    rr = pose.predict(im, imgsz=1280, conf=0.40, verbose=False, device=0)[0]
    if rr.keypoints is None or not len(rr.boxes): continue
    people = build_rois(rr.boxes.xyxy.cpu().numpy(), rr.keypoints.data.cpu().numpy())
    pb = [q.person_box for q in people]
    for pp in people:
        for at in ATT:
            hp = any(nm==at and centre_in(bb, pp.person_box) for bb,nm in gtb)
            hn = any(nm==f"no_{at}" and centre_in(bb, pp.person_box) for bb,nm in gtb)
            if hp == hn: continue
            truth = 0 if hp else 1                       # 1 = violation
            mp = any(nm==at and centre_in(bb, pp.person_box) for bb,nm in ppe)
            mn = any(nm==f"no_{at}" and centre_in(bb, pp.person_box) for bb,nm in ppe)
            mbg_pred = -1 if mp == mn else (0 if mp else 1)
            if at == "gloves":
                hs = []
                for h in pp.hands[:2]:
                    if a.gate and not roi_usable(h, im.shape, part="hand", others=pb, self_box=pp.person_box):
                        continue
                    c_ = crop_np(im, h, GEOMETRY.hand_crop_px)
                    if c_ is not None: hs.append(c_)
                if len(hs) < 2: eff_pred = -1
                else:
                    ps = eff_prob("gloves", hs[:2]); eff_pred = 0 if min(ps) >= .5 else 1
            else:
                gated = a.gate and not roi_usable(pp.head, im.shape, part="head", others=pb, self_box=pp.person_box)
                hc = None if (gated or pp.head is None) else crop_np(im, pp.head, GEOMETRY.head_crop_px)
                if hc is None: eff_pred = -1
                else: eff_pred = 0 if eff_prob(at, [hc])[0] >= .5 else 1
            rows[at].append((truth, eff_pred, mbg_pred))
    if n % 40 == 0: print(f"  {n}/{len(imgs)}", flush=True)

def score(pairs, operational):
    y = np.array([t for t, _ in pairs]); p = np.array([q for _, q in pairs])
    if operational:
        p = np.where(p < 0, 0, p)                 # abstain = no alert
        keep = np.ones(len(y), bool)
    else:
        keep = p >= 0
    if keep.sum() == 0 or len(set(y[keep])) < 2: return None
    pr, rc, f1, _ = precision_recall_fscore_support(y[keep], p[keep], average="binary", zero_division=0)
    return pr, rc, f1, int(keep.sum()), int(y[keep].sum())

report = {}
for at in ATT:
    r = rows[at]
    tot = len(r); viol = sum(1 for t,_,_ in r if t == 1)
    eff_ab = sum(1 for _,e,_ in r if e < 0); mbg_ab = sum(1 for _,_,m in r if m < 0)
    print(f"\n{at}: {tot} person-instances ({viol} violations)   "
          f"abstained: effnet-b1 {eff_ab} ({eff_ab/tot:.0%}), apd_mbg {mbg_ab} ({mbg_ab/tot:.0%})")
    print(f"  {'model':16} {'view':12} {'prec':>7} {'recall':>7} {'F1':>7} {'n':>6}")
    report[at] = {"n": tot, "violations": viol, "abstain": {"effnet": eff_ab, "apd_mbg": mbg_ab}, "scores": {}}
    for name, col in (("effnet-b1", 1), ("apd_mbg", 2)):
        pairs = [(t, (e if col == 1 else m)) for t, e, m in r]
        for view, op in (("committed", False), ("operational", True)):
            s = score(pairs, op)
            if s is None: continue
            pr, rc, f1, nn, nv = s
            report[at]["scores"][f"{name}/{view}"] = {"precision": pr, "recall": rc, "f1": f1, "n": nn}
            print(f"  {name:16} {view:12} {pr:7.1%} {rc:7.1%} {f1:7.3f} {nn:6d}")
Path(a.out).write_text(json.dumps(report, indent=2))
print(f"\nwrote {a.out}")
