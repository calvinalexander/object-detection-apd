"""Retrain EfficientNet-B1 with the evaluation data actually held out.

20% of head_cap and 34% of mask training crops were derived from apd_v3 VAL
images -- the split used to declare these models "measured on data no classifier
trained on". They were. Gloves was clean, which is why gloves is the one
attribute whose earlier numbers survive.

Excluded here:
  * every crop derived from an apd_v3 val image (the benchmark)
  * every stem in data/splits/frozen_test.json, plus its margin quarantine
    (the cross-model test set, so this model can be scored on it later)

No training script existed for the published EfficientNet-B1 -- it was trained
ad-hoc -- so this also makes the recipe reproducible.
"""
import argparse, glob, json, re, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/crops_ci_inpaint")
ap.add_argument("--attrs", default="head_cap,mask,gloves")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--out", default="runs/effnet_clean")
a = ap.parse_args()

VAL_IMGS = {p.stem for p in Path("data/dump/dataset/apd_v3/dataset/images/val").iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
frozen = json.loads(Path("data/splits/frozen_test.json").read_text())
HELD = set(frozen["test_stems"]) | set(frozen["quarantine_stems"])

def stem_of(p):
    s = Path(p).stem
    for suf in ("__o", "__h", "__g"):
        s = s.split(suf)[0]
    return s

def from_val_image(s):
    return s in VAL_IMGS or re.sub(r"_\d+(_h\d)?$", "", s) in VAL_IMGS

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

class Crops(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = cv2.imread(p)
        im = cv2.resize(im, (a.px, a.px))
        if self.train:
            if np.random.rand() < 0.5: im = im[:, ::-1]
            hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + np.random.randint(0, 180)) % 180      # hue is not evidence
            hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.7, 1.25), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.7, 1.3), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if np.random.rand() < 0.25:                                        # keep shape readable
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.tensor(float(y))

dev = "cuda"
Path(a.out).mkdir(parents=True, exist_ok=True)
summary = {}
for attr in a.attrs.split(","):
    tr, va, dropped = [], [], {"apd_v3_val": 0, "frozen_test": 0}
    for split, bucket in (("train", tr), ("val", va)):
        for cls, y in (("present", 1), ("absent", 0)):
            for p in glob.glob(f"{a.data}/{attr}/{split}/{cls}/*.jpg"):
                s = stem_of(p)
                if from_val_image(s): dropped["apd_v3_val"] += 1; continue
                if s in HELD: dropped["frozen_test"] += 1; continue
                bucket.append((p, y))
    if not tr or not va:
        print(f"{attr}: nothing left after exclusions"); continue
    pos = sum(y for _, y in tr)
    print(f"\n{attr}: train {len(tr)} ({pos} present / {len(tr)-pos} absent), val {len(va)}   "
          f"dropped {dropped['apd_v3_val']} apd_v3-val-derived, {dropped['frozen_test']} frozen-test")

    net = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 1)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    pw = torch.tensor([max(0.25, min(4.0, (len(tr)-pos)/max(1, pos)))], device=dev)
    dl_tr = DataLoader(Crops(tr, True), batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(Crops(va, False), batch_size=a.batch, num_workers=4)
    best, t0 = -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x)[:, 0], y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval(); S, Y = [], []
        with torch.no_grad():
            for x, y in dl_va:
                S.append(torch.sigmoid(net(x.to(dev))[:, 0]).cpu().numpy()); Y.append(y.numpy())
        S, Y = np.concatenate(S), np.concatenate(Y)
        auc = roc_auc_score(Y, S) if len(set(Y)) > 1 else float("nan")
        if auc > best:
            best = auc
            torch.save(net.state_dict(), Path(a.out) / f"{attr}_best.pt")
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  val AUC {auc:.4f}  best {best:.4f}  ({time.time()-t0:.0f}s)")
    summary[attr] = {"best_val_auc": float(best), "train": len(tr), "val": len(va), "dropped": dropped}
    print(f"  -> {Path(a.out)/f'{attr}_best.pt'}  best val AUC {best:.4f}")

(Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=2))
