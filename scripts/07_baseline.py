"""First honest numbers: linear probe on frozen DINOv2 features.

With 117 labelled head crops in train, fine-tuning a backbone would memorise the
set and tell us nothing. A linear probe on frozen self-supervised features is the
appropriate tool at this sample size, and it establishes the floor that any
larger model has to beat once labels exist.

Reported per attribute and per split. `mask` is deliberately excluded -- 9
training positives cannot support a claim either way, and quoting a number for it
would be the same mistake as the 0.958 mAP this project already made once.
"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoImageProcessor, AutoModel

from apd.schema import Attribute, Value, ATTRIBUTE_PART

CROPS = Path("artifacts/crops")
DEVICE = "cuda"
MODEL_ID = "facebook/dinov2-base"

splits = json.loads(Path("artifacts/splits.json").read_text())["splits"]
frame_split = {f: k for k, v in splits.items() for f in v}
seed = [json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()]
index = [json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]

head_crop = {r["person_id"]: r["crop"] for r in index if r["part"] == "head"}
hand_by_slot = {}
for r in index:
    if r["part"] == "hand":
        hand_by_slot[(r["person_id"], int(r["crop"].rsplit("_h", 1)[1].split(".")[0]))] = r["crop"]

print(f"loading {MODEL_ID} ...", flush=True)
proc = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

@torch.no_grad()
def embed(paths, batch=64):
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(CROPS / p).convert("RGB") for p in paths[i:i + batch]]
        px = proc(images=ims, return_tensors="pt").to(DEVICE)
        out.append(model(**px).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768))

def dataset(attr: Attribute):
    """Rows of (crop_path, y, split) for one attribute, skipping UNCLEAR."""
    rows = []
    for r in seed:
        s = frame_split.get(r["frame"])
        v = r["labels"][attr.value]
        if ATTRIBUTE_PART[attr] == "head" and v == Value.UNCLEAR.value:
            continue
        y = int(v == Value.PRESENT.value)
        if ATTRIBUTE_PART[attr] == "head":
            c = head_crop.get(r["person_id"])
            if c: rows.append((c, y, s))
        else:
            # Per-hand labels: the classifier learns "is this hand gloved" and the
            # both-hands rule is applied afterwards as logic. Stamping the person
            # label onto every hand, as an earlier version did, mislabelled the
            # bare hand of every one-gloved worker.
            for h in r.get("hands", []):
                c = hand_by_slot.get((r["person_id"], h["hand"]))
                if c and h["value"] != Value.UNCLEAR.value:
                    rows.append((c, int(h["value"] == Value.PRESENT.value), s))
    return rows

print(f"\n{'attribute':<10}{'split':<20}{'n':>5}{'pos':>5}{'P':>7}{'R':>7}{'F1':>7}{'AUC':>7}")
for attr in (Attribute.HEAD_CAP, Attribute.GLOVES):
    rows = dataset(attr)
    X = embed([r[0] for r in rows])
    y = np.array([r[1] for r in rows]); sp = np.array([r[2] for r in rows])

    tr = sp == "train"
    if tr.sum() < 20 or len(set(y[tr])) < 2:
        print(f"{attr.value:<10}insufficient training labels"); continue
    clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(X[tr], y[tr])

    for split in ("train", "val", "test_unseen_camera", "test_same_camera"):
        m = sp == split
        if m.sum() == 0 or len(set(y[m])) < 2:
            print(f"{attr.value:<10}{split:<20}{int(m.sum()):>5}{'-':>5}{'  (one class only)':>28}")
            continue
        pr = clf.predict(X[m]); pp = clf.predict_proba(X[m])[:, 1]
        print(f"{attr.value:<10}{split:<20}{int(m.sum()):>5}{int(y[m].sum()):>5}"
              f"{precision_score(y[m],pr,zero_division=0):>7.2f}"
              f"{recall_score(y[m],pr,zero_division=0):>7.2f}"
              f"{f1_score(y[m],pr,zero_division=0):>7.2f}"
              f"{roc_auc_score(y[m],pp):>7.2f}")

    # baseline to beat: predict the training-majority class everywhere
    m = sp == "test_unseen_camera"
    if m.sum() and len(set(y[m])) > 1:
        maj = int(y[tr].mean() > 0.5)
        print(f"{'':<10}{'  majority baseline':<20}{int(m.sum()):>5}{int(y[m].sum()):>5}"
              f"{precision_score(y[m],np.full(m.sum(),maj),zero_division=0):>7.2f}"
              f"{recall_score(y[m],np.full(m.sum(),maj),zero_division=0):>7.2f}"
              f"{f1_score(y[m],np.full(m.sum(),maj),zero_division=0):>7.2f}{'-':>7}")
    print()

print("mask: excluded -- 9 training positives cannot support a number.")

# ---------------------------------------------------------------------------
# How much can these numbers actually support? Bootstrap the headline split.
# ---------------------------------------------------------------------------
print("\nBOOTSTRAP on test_unseen_camera (2000 resamples, 95% interval)")
print(f"{'attribute':<10}{'n':>5}{'F1':>18}{'AUC':>18}")
rng = np.random.default_rng(0)
for attr in (Attribute.HEAD_CAP, Attribute.GLOVES):
    rows = dataset(attr)
    X = embed([r[0] for r in rows])
    y = np.array([r[1] for r in rows]); sp = np.array([r[2] for r in rows])
    tr = sp == "train"; te = sp == "test_unseen_camera"
    clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(X[tr], y[tr])
    yt = y[te]; pr = clf.predict(X[te]); pp = clf.predict_proba(X[te])[:, 1]
    f1s, aucs = [], []
    for _ in range(2000):
        idx = rng.integers(0, len(yt), len(yt))
        if len(set(yt[idx])) < 2:
            continue
        f1s.append(f1_score(yt[idx], pr[idx], zero_division=0))
        aucs.append(roc_auc_score(yt[idx], pp[idx]))
    lo_f, hi_f = np.percentile(f1s, [2.5, 97.5]); lo_a, hi_a = np.percentile(aucs, [2.5, 97.5])
    print(f"{attr.value:<10}{len(yt):>5}"
          f"{f'{np.mean(f1s):.2f} [{lo_f:.2f}-{hi_f:.2f}]':>18}"
          f"{f'{np.mean(aucs):.2f} [{lo_a:.2f}-{hi_a:.2f}]':>18}")
