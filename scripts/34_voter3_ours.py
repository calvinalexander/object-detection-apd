"""Voter 3: our pose->ROI->probe method, retrained on the production annotations.

Two things make this better supervision than our SPPG-derived labels:

  explicit negatives  `no_head_cap` / `no_mask` / `no_gloves` are annotated, so
      ABSENT is a positive assertion rather than "no box was drawn", which is the
      ambiguity that made the SPPG labels unreliable.

  per-hand gloves     a `gloves` box in one hand ROI and a `no_gloves` box in the
      other labels each hand independently, which is what the both-hands rule
      needs and what our earlier person-level derivation got wrong.

Split is camera-grouped (from 33_build_voter3_dataset), so the holdout measures
generalisation rather than memorised cameras.
"""
import collections, json, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.pose import build_rois, frontality
from apd.schema import GEOMETRY

POS = {"head_cap":2, "mask":3, "gloves":1}
NEG = {"head_cap":5, "mask":6, "gloves":4}
DEV, MID = "cuda", "facebook/dinov2-base"
ROOT = Path("data/voter3")
OUT = Path("artifacts/voter3"); OUT.mkdir(parents=True, exist_ok=True)

pose = YOLO(".work/yolo11x-pose.pt")
proc = AutoImageProcessor.from_pretrained(MID)
enc = AutoModel.from_pretrained(MID).to(DEV).eval()

@torch.no_grad()
def embed(ims, b=64):
    o=[]
    for i in range(0,len(ims),b):
        o.append(enc(**proc(images=ims[i:i+b],return_tensors="pt").to(DEV)).last_hidden_state[:,0].float().cpu().numpy())
    return np.concatenate(o) if o else np.zeros((0,768),np.float32)

def crop_of(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else Image.fromarray(cv2.resize(c,(size,size)))

def inside(box, roi):
    cx, cy = box[0]+box[2]/2, box[1]+box[3]/2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]

data = {s: {k: ([],[]) for k in POS} for s in ("train","val")}
stats = collections.Counter()
for split in ("train","val"):
    for img_p in sorted((ROOT/"images"/split).iterdir()):
        if img_p.suffix.lower() not in (".jpg",".jpeg",".png"): continue
        bgr = cv2.imread(str(img_p)); H,W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = collections.defaultdict(list)
        lab = ROOT/"labels"/split/(img_p.stem+".txt")
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            f = line.split()
            if len(f) < 5: continue
            c = int(f[0]); cx,cy,bw,bh = (float(v) for v in f[1:5])
            gt[c].append([(cx-bw/2)*W, (cy-bh/2)*H, bw*W, bh*H])

        r = pose.predict(bgr, imgsz=GEOMETRY.pose_imgsz, conf=0.25, verbose=False, device=0)[0]
        if r.keypoints is None or not len(r.boxes): continue
        kp = r.keypoints.data.cpu().numpy()
        people = build_rois(r.boxes.xyxy.cpu().numpy(), kp)
        frs = [frontality(k) for k in kp]

        for i, p in enumerate(people):
            if p.head is not None and not p.head_collision:
                c = crop_of(rgb, p.head, GEOMETRY.head_crop_px)
                if c is not None:
                    for attr in ("head_cap","mask"):
                        if attr == "mask" and frs[i] < 0.5:
                            stats["mask_skipped_not_frontal"] += 1; continue
                        hp = any(inside(b, p.head) for b in gt[POS[attr]])
                        hn = any(inside(b, p.head) for b in gt[NEG[attr]])
                        if hp == hn: stats[f"{attr}_no_evidence"] += 1; continue
                        data[split][attr][0].append(c); data[split][attr][1].append(int(hp))
            for hb in p.hands[:2]:
                c = crop_of(rgb, hb, GEOMETRY.hand_crop_px)
                if c is None: continue
                hp = any(inside(b, hb) for b in gt[POS["gloves"]])
                hn = any(inside(b, hb) for b in gt[NEG["gloves"]])
                if hp == hn: stats["gloves_no_evidence"] += 1; continue
                data[split]["gloves"][0].append(c); data[split]["gloves"][1].append(int(hp))

print("crops derived from the production annotations")
print(f"  {'attribute':<12}{'train +/-':>14}{'holdout +/-':>14}")
for k in POS:
    tr = np.array(data["train"][k][1]); ho = np.array(data["val"][k][1])
    print(f"  {k:<12}{f'{int(tr.sum())}/{len(tr)-int(tr.sum())}':>14}"
          f"{f'{int(ho.sum())}/{len(ho)-int(ho.sum())}':>14}")
print(f"  skipped for lack of evidence: {dict(stats)}\n")

print(f"{'attribute':<12}{'n':>5}{'AUC':>7}{'AUC 95%':>15}{'F1':>7}{'P':>6}{'R':>6}{'base':>7}")
rng = np.random.default_rng(0); models = {}
for k in POS:
    Xtr = embed(data["train"][k][0]); ytr = np.array(data["train"][k][1])
    Xho = embed(data["val"][k][0]);   yho = np.array(data["val"][k][1])
    if len(set(ytr)) < 2 or len(set(yho)) < 2:
        print(f"{k:<12}{len(yho):>5}   one class only"); continue
    clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(Xtr, ytr)
    models[k] = clf
    p = clf.predict_proba(Xho)[:,1]; pred = (p>=0.5).astype(int)
    auc = roc_auc_score(yho, p)
    a = [roc_auc_score(yho[i], p[i]) for i in (rng.integers(0,len(yho),len(yho)) for _ in range(2000))
         if len(set(yho[i]))>1]
    base = max(yho.mean(), 1-yho.mean())
    print(f"{k:<12}{len(yho):>5}{auc:>7.3f}"
          f"{f'[{np.percentile(a,2.5):.2f},{np.percentile(a,97.5):.2f}]':>15}"
          f"{f1_score(yho,pred):>7.2f}{precision_score(yho,pred,zero_division=0):>6.2f}"
          f"{recall_score(yho,pred):>6.2f}{base:>7.2f}")

import pickle
pickle.dump(models, open(OUT/"probes.pkl","wb"))
print(f"\nprobes -> {OUT}/probes.pkl   ('base' = always guessing the commoner class)")
