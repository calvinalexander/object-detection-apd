"""Score the student on the production annotation, alongside its teacher.

The student's own val split carries consensus labels, which the teacher wrote --
scoring the teacher there measures how faithfully the student copies it, not
whether either is right. The production val is annotated independently of both,
so it is the only place the comparison means anything.

Reported as accuracy where the model committed, plus commitment rate, so a model
cannot look good by abstaining on everything hard.
"""
import argparse, collections, json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision, torch.nn as nn
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from apd.pose import build_rois
from apd.schema import GEOMETRY, assert_contract

assert_contract()
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ATT = ["head_cap", "mask", "gloves"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dump/dataset/apd_v3/dataset")
ap.add_argument("--ckpt", default="runs/student/best.pt")
ap.add_argument("--margin", type=float, default=0.25)
a = ap.parse_args()

sys.path.insert(0, "scripts")
from importlib import import_module
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
PX = ck["px"]

class Student(nn.Module):
    def __init__(self, arch):
        super().__init__()
        b = getattr(torchvision.models, arch)(weights=None)
        self.stem = nn.Sequential(*list(b.children())[:-1])
        d = b.fc.in_features
        self.head_out = nn.Linear(d, 2); self.hand_out = nn.Linear(d, 1)
    def forward(self, head, h0, h1):
        f = self.stem(torch.cat([head, h0, h1], 0)).flatten(1); n = head.shape[0]
        return torch.cat([self.head_out(f[:n]),
                          torch.minimum(self.hand_out(f[n:2*n]), self.hand_out(f[2*n:]))], 1)

dev = "cuda"
net = Student(ck["arch"]).to(dev).eval(); net.load_state_dict(ck["model"])
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
def norm(im):
    im = cv2.resize(im, (PX, PX))
    x = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
    return torch.from_numpy(x.transpose(2,0,1).copy())[None]

pose = YOLO(".work/yolo11x-pose.pt")
probes = pickle.load(open("artifacts/voter3_ci3/probes.pkl","rb"))
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
enc = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
y26 = {k: YOLO(f"runs/classify/runs/classify/runs/ci3_{k}/weights/best.pt") for k in ATT}
y26i = {k: {v:i for i,v in m.names.items()} for k,m in y26.items()}

@torch.no_grad()
def embed(ims):
    if not ims: return np.zeros((0,768), np.float32)
    return enc(**proc(images=ims, return_tensors="pt").to(dev)).last_hidden_state[:,0].float().cpu().numpy()

def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

def centre_in(b,p):
    cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
    return p[0]<=cx<=p[2] and p[1]<=cy<=p[3]

root = Path(a.data)
imgs = sorted(p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
stat = collections.defaultdict(lambda: collections.Counter())
raw = collections.defaultdict(list)
for n, p in enumerate(imgs, 1):
    bgr = cv2.imread(str(p))
    if bgr is None: continue
    H, W = bgr.shape[:2]
    gtb = []
    lp = root/"labels/val"/(p.stem+".txt")
    if lp.exists():
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) < 5: continue
            c = int(f[0]); cx,cy,bw,bh = (float(v) for v in f[1:5])
            gtb.append(([(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H], NAMES[c]))
    rr = pose.predict(bgr, imgsz=1280, conf=0.40, verbose=False, device=0)[0]
    if rr.keypoints is None or not len(rr.boxes): continue
    kp = rr.keypoints.data.cpu().numpy()
    people = build_rois(rr.boxes.xyxy.cpu().numpy(), kp)
    for pp in people:
        hc = crop_np(bgr, pp.head, GEOMETRY.head_crop_px) if (pp.head is not None and not pp.head_collision) else None
        hands = [c_ for c_ in (crop_np(bgr, h, GEOMETRY.hand_crop_px) for h in pp.hands[:2]) if c_ is not None]
        truth = {}
        for at in ATT:
            hp = any(nm==at and centre_in(bb, pp.person_box) for bb,nm in gtb)
            hn = any(nm==f"no_{at}" and centre_in(bb, pp.person_box) for bb,nm in gtb)
            if hp != hn: truth[at] = "present" if hp else "absent"
        if not truth: continue

        # --- student
        with torch.no_grad():
            h_t = norm(hc).to(dev) if hc is not None else torch.zeros(1,3,PX,PX, device=dev)
            hs = [norm(c_).to(dev) for c_ in hands[:2]]
            while len(hs) < 2: hs.append(torch.zeros(1,3,PX,PX, device=dev))
            s = torch.sigmoid(net(h_t, hs[0], hs[1]))[0].cpu().numpy()
        for j, at in enumerate(ATT):
            if at not in truth: continue
            avail = (hc is not None) if at != "gloves" else (len(hands) == 2)
            if avail:
                raw[at].append((float(s[j]), 1 if truth[at] == "present" else 0))
            if not avail or abs(s[j]-0.5) < a.margin:
                stat[f"{at}:student"]["abstain"] += 1
            else:
                stat[f"{at}:student"]["n"] += 1
                stat[f"{at}:student"]["ok"] += int(("present" if s[j] >= .5 else "absent") == truth[at])

        # --- teacher (voter3 probes + voter4 ci3), same abstention rules
        e = embed([Image.fromarray(cv2.cvtColor(hc, cv2.COLOR_BGR2RGB))])[0] if hc is not None else None
        for at in ATT:
            if at not in truth: continue
            if at == "gloves":
                if len(hands) < 2:
                    stat[f"{at}:teacher"]["abstain"] += 1; continue
                eh = embed([Image.fromarray(cv2.cvtColor(c_, cv2.COLOR_BGR2RGB)) for c_ in hands[:2]])
                p3 = [float(probes["gloves"].predict_proba(x[None])[0,1]) for x in eh]
                p4 = [float(y26["gloves"].predict(c_, verbose=False, device=0)[0]
                      .probs.data[y26i["gloves"]["present"]]) for c_ in hands[:2]]
                v3 = "present" if all(x>=.5 for x in p3) else "absent"
                v4 = "present" if all(x>=.5 for x in p4) else "absent"
            else:
                if hc is None:
                    stat[f"{at}:teacher"]["abstain"] += 1; continue
                q3 = float(probes[at].predict_proba(e[None])[0,1])
                q4 = float(y26[at].predict(hc, verbose=False, device=0)[0].probs.data[y26i[at]["present"]])
                if at == "mask" and (abs(q3-.5) < a.margin or abs(q4-.5) < a.margin):
                    stat[f"{at}:teacher"]["abstain"] += 1; continue
                v3 = "present" if q3>=.5 else "absent"
                v4 = "present" if q4>=.5 else "absent"
            if v3 != v4:
                stat[f"{at}:teacher"]["abstain"] += 1
            else:
                stat[f"{at}:teacher"]["n"] += 1
                stat[f"{at}:teacher"]["ok"] += int(v3 == truth[at])
    if n % 40 == 0: print(f"  {n}/{len(imgs)}", flush=True)

from sklearn.metrics import roc_auc_score
print(f"\nstudent, threshold-free and at matched coverage ({a.ckpt})")
print(f"{'attribute':10} {'n':>5} {'AUC':>7} {'acc@70% cov':>13} {'acc calibrated':>15}")
for at in ATT:
    v = raw[at]
    if len(v) < 10 or len({y for _, y in v}) < 2: continue
    sc = np.array([x for x, _ in v]); yy = np.array([y for _, y in v])
    auc = roc_auc_score(yy, sc)
    # keep the 70% most confident, so coverage is equal across models
    conf = np.abs(sc - 0.5); thr = np.quantile(conf, 0.30); k = conf >= thr
    acc = (((sc[k] >= .5).astype(int)) == yy[k]).mean()
    # A soft-distilled model is compressed toward the teacher's mean, so 0.5 is
    # not its operating point. Pick the threshold on one half, apply to the
    # other, so the number is not the threshold fitted to what it scores.
    half = len(sc) // 2
    idx = np.argsort(np.arange(len(sc)) % 2)      # deterministic interleave
    fit, hold = idx[:half], idx[half:]
    best_t, best_a = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        aa = (((sc[fit] >= t).astype(int)) == yy[fit]).mean()
        if aa > best_a: best_a, best_t = aa, t
    acc_cal = (((sc[hold] >= best_t).astype(int)) == yy[hold]).mean()
    print(f"{at:10} {len(v):5d} {auc:7.3f} {acc:12.1%} {acc_cal:14.1%}  (t={best_t:.2f})")

print(f"\nproduction val ({len(imgs)} images), margin {a.margin}")
print(f"{'attribute':10} {'model':9} {'committed':>18} {'correct':>9}")
for at in ATT:
    for m in ("teacher", "student"):
        c = stat[f"{at}:{m}"]
        tot = c["n"] + c["abstain"]
        if not tot: continue
        print(f"{at:10} {m:9} {c['n']:6d}/{tot:<5d} ({c['n']/tot:5.1%}) {c['ok']/max(1,c['n']):8.1%}")
