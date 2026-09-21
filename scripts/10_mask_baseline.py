"""Retrain the mask probe on seed + mined labels and report what it can support."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoImageProcessor, AutoModel
from apd.schema import Attribute, Value

CROPS = Path("artifacts/crops"); DEVICE, MODEL_ID = "cuda", "facebook/dinov2-base"
splits = json.loads(Path("artifacts/splits.json").read_text())["splits"]
frame_split = {f: k for k, v in splits.items() for f in v}
head_crop = {r["person_id"]: r["crop"] for r in (json.loads(l) for l in
             Path("artifacts/crops_index.jsonl").read_text().splitlines()) if r["part"] == "head"}

rows = {}
for r in (json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()):
    v = r["labels"][Attribute.MASK.value]
    if v != Value.UNCLEAR.value:
        rows[r["person_id"]] = (v, frame_split.get(r["frame"]), "seed")
for m in (json.loads(l) for l in Path("artifacts/mined_labels.jsonl").read_text().splitlines()):
    if m["attribute"] == Attribute.MASK.value and m["value"] != Value.UNCLEAR.value:
        rows.setdefault(m["person_id"], (m["value"], m["split"], "mined"))

items = [(head_crop[p], int(v == "present"), s, src)
         for p, (v, s, src) in rows.items() if p in head_crop]
proc = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

@torch.no_grad()
def embed(paths, batch=96):
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(CROPS / p).convert("RGB") for p in paths[i:i + batch]]
        out.append(model(**proc(images=ims, return_tensors="pt").to(DEVICE)
                         ).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out)

X = embed([i[0] for i in items])
y = np.array([i[1] for i in items]); sp = np.array([i[2] for i in items])
tr = sp == "train"
print(f"train: {int(y[tr].sum())} positives / {int(len(y[tr]) - y[tr].sum())} negatives")
clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(X[tr], y[tr])

print(f"\n{'split':<20}{'n':>5}{'pos':>5}{'P':>7}{'R':>7}{'F1':>7}{'AUC':>7}")
for split in ("train", "val", "test_unseen_scene", "test_same_camera"):
    m = sp == split
    if m.sum() == 0 or len(set(y[m])) < 2:
        print(f"{split:<20}{int(m.sum()):>5}{int(y[m].sum()) if m.sum() else 0:>5}   one class only")
        continue
    pr, pp = clf.predict(X[m]), clf.predict_proba(X[m])[:, 1]
    print(f"{split:<20}{int(m.sum()):>5}{int(y[m].sum()):>5}"
          f"{precision_score(y[m],pr,zero_division=0):>7.2f}{recall_score(y[m],pr,zero_division=0):>7.2f}"
          f"{f1_score(y[m],pr,zero_division=0):>7.2f}{roc_auc_score(y[m],pp):>7.2f}")

m = sp == "test_unseen_scene"
rng = np.random.default_rng(0); pr, pp = clf.predict(X[m]), clf.predict_proba(X[m])[:, 1]
f1s = [f1_score(y[m][i], pr[i], zero_division=0) for i in
       (rng.integers(0, m.sum(), m.sum()) for _ in range(2000)) if len(set(y[m][i])) > 1]
lo, hi = np.percentile(f1s, [2.5, 97.5])
print(f"\ntest_unseen_scene F1 {np.mean(f1s):.2f} [{lo:.2f}-{hi:.2f}] on {int(y[m].sum())} positives")
print("The interval is this wide because the TEST set was never mined -- deliberately,")
print("since ranking a held-out split would destroy its prevalence. Training is unblocked;")
print("evaluation is not, and needs a random-sampled labelling pass of its own.")
