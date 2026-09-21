"""Compare every classifier generation on one set none of them trained on.

The `ci3` vs `ci_inpaint` table in PROGRESS.md compared two models on two
different validation splits that shared 55-67% of their crops with each other's
TRAINING data. That is the fourth leaky split in this project, so this script
builds the evaluation set by exclusion instead of trusting any existing one:

  * a crop is eligible only if its stem appears in NO training directory of any
    model being compared
  * and only if its SITE appears in no training directory either, so a model
    cannot score well by having memorised the room
  * labels are the current post-sanitation ones; the hat relabelling moved 345
    crops, so an older label file would score today's models against a contract
    they were not trained on

Every model sees identical pixels (the inpainted crop), because that is what a
deployed model gets once overlay removal runs. Small n is reported honestly with
a bootstrap interval rather than a bare point estimate.
"""
import argparse, collections, glob, json, random, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO
from apd.schema import Attribute, assert_contract

assert_contract()
ATTRS = [a.value for a in Attribute]
ap = argparse.ArgumentParser()
ap.add_argument("--truth", default="data/crops_ci_inpaint")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", default="artifacts/fair_compare.json")
a = ap.parse_args()

TRAIN_DIRS = ["data/crops_ci_inpaint", "data/crops_ci3", "data/crops_cls",
              "data/crops_ci", "data/crops_nox"]

def stem_of(p):
    """Strip the augmentation suffix so `x__h45` and `x__o` are one crop."""
    s = Path(p).stem
    return s.split("__o")[0].split("__h")[0].split("__g")[0]

def site_of(stem):
    return stem.split("__")[0] if "__" in stem else "sppg"

MODELS = {
    "ci3 (yolo26n-cls)":        ("yolo", "runs/classify/runs/classify/runs/ci3_%s/weights/best.pt"),
    "inpaint (yolo11n-cls)":    ("yolo", "runs/classify/runs/classify_inpaint/%s/weights/best.pt"),
    "inpaint (yolo11s-cls)":    ("yolo", "runs/classify/runs/classify_sm_experiment/yolo11s-cls_%s/weights/best.pt"),
    "inpaint (yolo11m-cls)":    ("yolo", "runs/classify/runs/classify_sm_experiment/yolo11m-cls_%s/weights/best.pt"),
    "inpaint (effnet-b1)":      ("effnet", "runs/classify_effnet_b1/%s_best.pt"),
    "voter4 (original)":        ("yolo", "runs/classify/runs/voter4_%s/weights/best.pt"),
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_effnet(path):
    net = torchvision.models.efficientnet_b1(weights=None)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    n_out = sd["classifier.1.weight"].shape[0]
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, n_out)
    net.load_state_dict(sd)
    return net.cuda().eval(), n_out


def score(kind, path, paths):
    if kind == "yolo":
        m = YOLO(path)
        idx = [k for k, v in m.names.items() if v == "present"][0]
        out = []
        for i in range(0, len(paths), 64):
            for r in m.predict([str(p) for p in paths[i:i+64]], imgsz=128,
                               verbose=False, device=0):
                out.append(float(r.probs.data[idx]))
        return np.array(out)
    net, n_out = load_effnet(path)
    out = []
    for i in range(0, len(paths), 64):
        b = []
        for p in paths[i:i+64]:
            im = cv2.resize(cv2.imread(str(p)), (128, 128))
            x = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
            b.append(x.transpose(2, 0, 1))
        with torch.no_grad():
            lo = net(torch.tensor(np.array(b)).cuda())
            pr = torch.softmax(lo, 1)[:, 1] if n_out == 2 else torch.sigmoid(lo[:, 0])
        out.append(pr.cpu().numpy())
    return np.concatenate(out)


def boot_ci(y, s, n):
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k])) < 2:
            continue
        vals.append(roc_auc_score(y[k], s[k]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),)*2


report = {}
for attr in ATTRS:
    # ---- everything any model trained on, by stem and by site
    train_stems, train_sites = set(), set()
    for d in TRAIN_DIRS:
        for p in glob.glob(f"{d}/{attr}/train/*/*.jpg"):
            st = stem_of(p); train_stems.add(st); train_sites.add(site_of(st))

    items = []
    for cls in ("present", "absent"):
        for p in glob.glob(f"{a.truth}/{attr}/val/{cls}/*.jpg"):
            st = stem_of(p)
            if st in train_stems or site_of(st) in train_sites:
                continue
            items.append((p, 1 if cls == "present" else 0, site_of(st)))

    y = np.array([t for _, t, _ in items])
    paths = [p for p, _, _ in items]
    sites = {s for _, _, s in items}
    print(f"\n{attr}: {len(items)} eligible crops "
          f"({int(y.sum())} present / {int((1-y).sum())} absent) from {len(sites)} unseen sites")
    if len(items) < 12 or len(set(y)) < 2:
        print("  too few to score — reported as unmeasurable, not as a number")
        report[attr] = {"n": len(items), "status": "unmeasurable"}
        continue

    report[attr] = {"n": len(items), "present": int(y.sum()), "sites": len(sites), "models": {}}
    print(f"  {'model':26} {'AUC':>6}  {'95% CI':>16} {'acc':>7}")
    for name, (kind, tmpl) in MODELS.items():
        path = tmpl % attr
        if not Path(path).exists():
            continue
        try:
            s = score(kind, path, paths)
        except Exception as e:
            print(f"  {name:26}  failed: {type(e).__name__}")
            continue
        auc = roc_auc_score(y, s); lo, hi = boot_ci(y, s, a.boot)
        acc = ((s >= 0.5).astype(int) == y).mean()
        report[attr]["models"][name] = {"auc": float(auc), "ci": [lo, hi], "acc": float(acc)}
        print(f"  {name:26} {auc:6.3f}  [{lo:.3f}, {hi:.3f}] {acc:7.1%}")

Path(a.out).write_text(json.dumps(report, indent=2))
print(f"\nwrote {a.out}")
