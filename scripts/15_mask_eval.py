"""Mask evaluation on the Opus-labelled held-out split.

Label sources are merged in precedence order -- seed boxes, then mined, then the
bulk pass -- and every one of them is re-checked against the geometric gate,
because 10 of the 28 mined labels were selected while the gate was broken and
cannot be trusted now.
"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoImageProcessor, AutoModel
from apd.pose import frontality

CROPS = Path("artifacts/crops"); DEVICE, MODEL_ID = "cuda", "facebook/dinov2-base"
persons = {p["person_id"]: p for p in (json.loads(l) for l in
           Path("artifacts/persons.jsonl").read_text().splitlines())}
head = {r["person_id"]: r["crop"] for r in (json.loads(l) for l in
        Path("artifacts/crops_index.jsonl").read_text().splitlines()) if r["part"] == "head"}
splits = json.loads(Path("artifacts/splits.json").read_text())["splits"]
fs = {f: k for k, v in splits.items() for f in v}

def gated(pid):
    p = persons.get(pid)
    return bool(p and p["head_roi"] and not p["head_collision"]
                and p["person_conf"] >= 0.35 and frontality(p["kp"]) >= 0.5)

rows, src = {}, collections.Counter()
for r in (json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()):
    v = r["labels"]["mask"]
    if v != "unclear" and gated(r["person_id"]):
        rows[r["person_id"]] = (v, fs.get(r["frame"])); src["seed"] += 1
for m in (json.loads(l) for l in Path("artifacts/mined_labels.jsonl").read_text().splitlines()):
    if m["attribute"] == "mask" and m["value"] != "unclear" and gated(m["person_id"]):
        rows.setdefault(m["person_id"], (m["value"], m["split"])); src["mined"] += 1
for b in (json.loads(l) for l in Path("artifacts/bulk_labels.jsonl").read_text().splitlines()):
    if b["value"] != "unclear" and gated(b["person_id"]):
        rows.setdefault(b["person_id"], (b["value"], b["split"])); src["bulk_opus"] += 1
print("label sources:", dict(src))

items = [(head[p], int(v == "present"), s) for p, (v, s) in rows.items() if p in head]
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

X = embed([i[0] for i in items]); y = np.array([i[1] for i in items])
sp = np.array([i[2] for i in items])
tr = sp == "train"
print(f"\ntrain: {int(y[tr].sum())} present / {int(len(y[tr]) - y[tr].sum())} absent")
clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(X[tr], y[tr])

print(f"\n{'split':<20}{'n':>5}{'pos':>5}{'P':>7}{'R':>7}{'F1':>7}{'AUC':>7}")
for split in ("train", "val", "test_unseen_camera", "test_same_camera"):
    m = sp == split
    if m.sum() == 0 or len(set(y[m])) < 2:
        print(f"{split:<20}{int(m.sum()):>5}{int(y[m].sum()) if m.sum() else 0:>5}   one class only")
        continue
    pr, pp = clf.predict(X[m]), clf.predict_proba(X[m])[:, 1]
    print(f"{split:<20}{int(m.sum()):>5}{int(y[m].sum()):>5}"
          f"{precision_score(y[m],pr,zero_division=0):>7.2f}{recall_score(y[m],pr,zero_division=0):>7.2f}"
          f"{f1_score(y[m],pr,zero_division=0):>7.2f}{roc_auc_score(y[m],pp):>7.2f}")

m = sp == "test_unseen_camera"
if m.sum() == 0 or len(set(y[m])) < 2 or int(y[m].sum()) < 5:
    print(f"\ntest_unseen_camera: {int(y[m].sum())} positives / {int(m.sum())} labels")
    print("  Too few to bootstrap. No number is reported rather than a meaningless one.")
    raise SystemExit
pr, pp = clf.predict(X[m]), clf.predict_proba(X[m])[:, 1]
rng = np.random.default_rng(0); f1s, aucs = [], []
for _ in range(2000):
    i = rng.integers(0, m.sum(), m.sum())
    if len(set(y[m][i])) < 2: continue
    f1s.append(f1_score(y[m][i], pr[i], zero_division=0)); aucs.append(roc_auc_score(y[m][i], pp[i]))
maj = int(y[tr].mean() > 0.5)
print(f"\ntest_unseen_camera, {int(y[m].sum())} positives / {int(m.sum())} labels")
print(f"  F1  {np.mean(f1s):.2f} [{np.percentile(f1s,2.5):.2f}-{np.percentile(f1s,97.5):.2f}]")
print(f"  AUC {np.mean(aucs):.2f} [{np.percentile(aucs,2.5):.2f}-{np.percentile(aucs,97.5):.2f}]")
print(f"  majority-class baseline F1: {f1_score(y[m], np.full(int(m.sum()), maj), zero_division=0):.2f}")
