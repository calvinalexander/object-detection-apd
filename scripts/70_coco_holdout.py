"""Train on everything except the human-annotated frames; evaluate on those.

The 53 COCO-annotated SPPG frames are the only human labels in this project.
Everything else -- site crops, apd_v3 val crops, the rest of the SPPG frames --
is model-generated or borrowed, so it belongs in training rather than in the
measurement. Crops derived from the 53 frames are excluded from training:
22% of head_cap, 9% of mask, 33% of gloves.

Person-level truth comes from artifacts/seed_labels.jsonl, which was derived
from the COCO boxes and is aligned with artifacts/persons.jsonl, so the same
person boxes and ROIs are used here as when the truth was written. `unclear` is
skipped, not scored.

One asymmetry is unavoidable and is stated rather than hidden: apd_mbg may have
seen these frames in its own training -- we do not have its data and cannot
check. Our model definitely has not.
"""
import argparse, collections, glob, json, re, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torchvision, openvino as ov
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.pose import roi_usable
from apd.schema import GEOMETRY, assert_contract

assert_contract()
ATTRS = ["head_cap", "mask", "gloves"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/crops_ci_inpaint")
ap.add_argument("--epochs", type=int, default=35)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--out", default="runs/effnet_coco_holdout")
ap.add_argument("--skip-train", type=int, default=0)
ap.add_argument("--mbg", default="model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml",
                help="which production build to score")
ap.add_argument("--gate", type=int, default=0,
                help="apply the ROI admissibility gate before classifying, so people "
                     "who cannot be read (tiny, clipped, occluded) are abstained on "
                     "rather than guessed at")
a = ap.parse_args()

coco = json.loads(Path("data/sppg/.exports/coco-1778210446.4251382.json").read_text())
imgs = {i["id"]: i["file_name"] for i in coco["images"]}
ANNOTATED = {Path(imgs[x["image_id"]]).stem for x in coco["annotations"]}

def stem_of(p):
    s = Path(p).stem
    for suf in ("__o", "__h", "__g"):
        s = s.split(suf)[0]
    return s
def from_annotated(s):
    return s in ANNOTATED or re.sub(r"_\d+(_h\d)?$", "", s) in ANNOTATED

MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
def norm(im):
    x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
    return x.transpose(2,0,1)

class Crops(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = cv2.resize(cv2.imread(p), (a.px, a.px))
        if np.random.rand() < 0.5: im = im[:, ::-1]
        hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[...,0] = (hsv[...,0] + np.random.randint(0,180)) % 180
        hsv[...,1] = np.clip(hsv[...,1]*np.random.uniform(0.7,1.25), 0, 255)
        hsv[...,2] = np.clip(hsv[...,2]*np.random.uniform(0.7,1.3), 0, 255)
        im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if np.random.rand() < 0.25:
            im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return torch.from_numpy(norm(im).copy()), torch.tensor(float(y))

dev = "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
def build(attr):
    net = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 1)
    return net.to(dev)

if not a.skip_train:
    for attr in ATTRS:
        items, dropped = [], 0
        for sp in ("train", "val"):
            for cls, y in (("present",1), ("absent",0)):
                for p in glob.glob(f"{a.data}/{attr}/{sp}/{cls}/*.jpg"):
                    if from_annotated(stem_of(p)): dropped += 1; continue
                    items.append((p, y))
        pos = sum(y for _, y in items)
        print(f"{attr}: train {len(items)} ({pos}+/{len(items)-pos}-), dropped {dropped} from annotated frames")
        net = build(attr)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        pw = torch.tensor([max(0.25, min(4.0, (len(items)-pos)/max(1,pos)))], device=dev)
        dl = DataLoader(Crops(items), batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
        t0 = time.time()
        for ep in range(1, a.epochs+1):
            net.train(); tot = 0.0
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x)[:,0], y, pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            sched.step()
            if ep % 10 == 0 or ep == a.epochs:
                print(f"  ep {ep:3d} loss {tot/len(dl):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        torch.save(net.state_dict(), out / f"{attr}_last.pt")   # no selection on the eval set

# ---------------- evaluate ----------------
persons = {json.loads(l)["person_id"]: json.loads(l) for l in Path("artifacts/persons.jsonl").read_text().splitlines()}
truth = collections.defaultdict(dict)
for line in Path("artifacts/seed_labels.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["person_id"] not in persons: continue
    for at, v in (r.get("labels") or {}).items():
        if v in ("present", "absent"): truth[at][r["person_id"]] = v

nets = {}
for attr in ATTRS:
    n = build(attr); n.load_state_dict(torch.load(out/f"{attr}_last.pt", map_location="cpu", weights_only=False))
    nets[attr] = n.eval()
def eff_p(attr, crops):
    with torch.no_grad():
        x = torch.tensor(np.array([norm(cv2.resize(c,(a.px,a.px))) for c in crops])).to(dev)
        return torch.sigmoid(nets[attr](x)[:,0]).cpu().numpy()

core = ov.Core(); dv = "GPU" if "GPU" in core.available_devices else "CPU"
mbg = core.compile_model(core.read_model(a.mbg), dv)
port = mbg.outputs[0]
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
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

rows = collections.defaultdict(list)
frames = sorted({persons[pid]["frame"] for at in ATTRS for pid in truth[at]})
for k, fname in enumerate(frames, 1):
    im = cv2.imread(str(Path("data/sppg")/fname))
    if im is None: continue
    img, r, dx, dy = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    o = mbg([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
    if o.shape[0] < o.shape[1]: o = o.T
    sc = o[:,4:11]; cl = sc.argmax(1); cf = sc.max(1); m = cf >= a.conf
    b, cf, cl = o[m][:,:4], cf[m], cl[m]; ppe = []
    if len(b):
        xy = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2],1)
        xy[:,[0,2]] -= dx; xy[:,[1,3]] -= dy; xy /= r
        idx = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                               cf.tolist(), a.conf, 0.7)
        idx = np.array(idx).flatten() if len(idx) else np.zeros(0,int)
        ppe = [(xy[i], NAMES[cl[i]]) for i in idx if NAMES[cl[i]] != "person"]
    frame_boxes = [persons[q]["person_box"] for at2 in ATTRS for q in truth[at2]
                   if persons[q]["frame"] == fname]
    for at in ATTRS:
        for pid, tv in truth[at].items():
            rec = persons[pid]
            if rec["frame"] != fname: continue
            pbox = rec["person_box"]
            y = 0 if tv == "present" else 1                     # 1 = violation
            mp = any(nm==at and centre_in(bb, pbox) for bb,nm in ppe)
            mn = any(nm==f"no_{at}" and centre_in(bb, pbox) for bb,nm in ppe)
            mbg_pred = -1 if mp == mn else (0 if mp else 1)
            if at == "gloves":
                hb = [h for h in (rec["hand_rois"] or [])[:2]
                      if not a.gate or roi_usable(h, im.shape, part="hand",
                                                  others=frame_boxes, self_box=pbox)]
                hs = [c for c in (crop_np(im, h, GEOMETRY.hand_crop_px) for h in hb) if c is not None]
                eff_pred = -1 if len(hs) < 2 else (0 if min(eff_p("gloves", hs[:2])) >= .5 else 1)
            else:
                ok = rec.get("head_roi") and (not a.gate or roi_usable(
                    rec["head_roi"], im.shape, part="head", others=frame_boxes, self_box=pbox))
                hc = crop_np(im, rec["head_roi"], GEOMETRY.head_crop_px) if ok else None
                eff_pred = -1 if hc is None else (0 if eff_p(at, [hc])[0] >= .5 else 1)
            rows[at].append((y, eff_pred, mbg_pred))
    if k % 15 == 0: print(f"  eval {k}/{len(frames)}", flush=True)

def extra(pairs):
    y = np.array([t for t,_ in pairs]); p = np.array([max(0,q) for _,q in pairs])
    mcc = matthews_corrcoef(y, p) if len(set(p)) > 1 else 0.0
    return mcc, balanced_accuracy_score(y, p)

def score(pairs, operational):
    y = np.array([t for t,_ in pairs]); p = np.array([q for _,q in pairs])
    if operational: p = np.where(p < 0, 0, p); keep = np.ones(len(y), bool)
    else: keep = p >= 0
    if keep.sum() == 0 or len(set(y[keep])) < 2: return None
    pr, rc, f1, _ = precision_recall_fscore_support(y[keep], p[keep], average="binary", zero_division=0)
    return pr, rc, f1, int(keep.sum())

print("\n=== human COCO annotation as truth (violation = positive) ===")
rep = {}
for at in ATTRS:
    r = rows[at]
    if not r: continue
    viol = sum(1 for t,_,_ in r if t == 1)
    ea = sum(1 for _,e,_ in r if e < 0); ma = sum(1 for _,_,m in r if m < 0)
    print(f"\n{at}: {len(r)} persons ({viol} violations)   abstain: ours {ea} ({ea/len(r):.0%}), apd_mbg {ma} ({ma/len(r):.0%})")
    bp = np.mean([t for t,_,_ in r])
    print(f"  {'model':18} {'view':12} {'prec':>7} {'recall':>7} {'F1':>7} {'MCC':>7} {'bal-acc':>8} {'n':>5}")
    print(f"  {'ALWAYS-FLAG':18} {'baseline':12} {bp:7.1%} {1.0:7.1%} {2*bp/(1+bp):7.3f} {0.0:7.3f} {0.5:8.1%} {len(r):5d}")
    rep[at] = {"n": len(r), "violations": viol, "scores": {}}
    for name, col in (("ours effnet-b1", 1), ("apd_mbg", 2)):
        for view, op in (("committed", False), ("operational", True)):
            s = score([(t, (e if col==1 else m)) for t,e,m in r], op)
            if s is None: continue
            pr, rc, f1, nn = s
            mcc, ba = extra([(t, (e if col==1 else m)) for t,e,m in r])
            rep[at]["scores"][f"{name}/{view}"] = dict(precision=pr, recall=rc, f1=f1, n=nn, mcc=mcc, bal_acc=ba)
            print(f"  {name:18} {view:12} {pr:7.1%} {rc:7.1%} {f1:7.3f} {mcc:7.3f} {ba:8.1%} {nn:5d}")
(out/"coco_eval.json").write_text(json.dumps(rep, indent=2))
print(f"\nwrote {out/'coco_eval.json'}")
