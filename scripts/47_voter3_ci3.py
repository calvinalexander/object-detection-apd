"""Voter 3 rebuilt: DINOv2 probes on the cleaned, site-diverse crops.

Same data and same site-disjoint splits as voter4's ci3 models, so the two
voters differ in representation (frozen DINOv2 + linear probe vs a fine-tuned
YOLO26 classifier) rather than in what they were shown. A vote is only worth
counting if the voters can disagree for reasons other than their training set.

Hue-rotated copies are kept in the probe's training data: the probe is linear,
so it cannot learn an invariance the features do not express, but it can be
stopped from leaning on a hue direction that is present in them.
"""
import glob, json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, assert_contract

assert_contract()
DEV, MID = "cuda", "facebook/dinov2-base"
DATA = Path("data/crops_ci3")
OUT = Path("artifacts/voter3_ci3"); OUT.mkdir(parents=True, exist_ok=True)

proc = AutoImageProcessor.from_pretrained(MID)
enc = AutoModel.from_pretrained(MID).to(DEV).eval()

@torch.no_grad()
def embed(paths, b=48):
    out = []
    for i in range(0, len(paths), b):
        ims = [Image.fromarray(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
               for p in paths[i:i + b]]
        out.append(enc(**proc(images=ims, return_tensors="pt").to(DEV))
                   .last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

probes, report = {}, {}
for attr in [a.value for a in Attribute]:
    d = DATA / attr
    if not d.is_dir():
        continue
    tr = {c: sorted(glob.glob(str(d / "train" / c / "*.jpg"))) for c in ("present", "absent")}
    va = {c: sorted(glob.glob(str(d / "val" / c / "*__o.jpg"))) for c in ("present", "absent")}
    Xtr = embed(tr["present"] + tr["absent"])
    ytr = np.r_[np.ones(len(tr["present"])), np.zeros(len(tr["absent"]))]
    Xva = embed(va["present"] + va["absent"])
    yva = np.r_[np.ones(len(va["present"])), np.zeros(len(va["absent"]))]

    clf = LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced").fit(Xtr, ytr)
    s = clf.predict_proba(Xva)[:, 1]
    auc, acc = roc_auc_score(yva, s), ((s > 0.5) == yva).mean()
    probes[attr] = clf
    report[attr] = {"train": len(ytr), "val": len(yva), "auc": float(auc), "acc": float(acc)}
    print(f"{attr:9} train {len(ytr):5d}  val {len(yva):4d}   AUC {auc:.3f}  acc {acc:.3f}")

pickle.dump(probes, open(OUT / "probes.pkl", "wb"))
(OUT / "report.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote {OUT/'probes.pkl'}")
