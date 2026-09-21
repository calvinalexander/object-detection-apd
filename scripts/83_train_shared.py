"""One backbone for all three attributes, instead of three trained separately.

head_cap and mask are learned from the SAME head crop -- 281 of 327 mask crops
are also head_cap crops -- so training them apart means learning "this is a head"
twice from scratch, and mask learns it from only 327 examples. A shared trunk
lets mask borrow head_cap's data for the representation and spend its own labels
on the mask-specific decision.

Gloves rides the same trunk on hand crops. A crop contributes loss only to the
heads its label covers: a head crop with no mask label trains head_cap alone.
"""
import argparse, glob, json, sys, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import assert_contract

assert_contract()
ATT = ["head_cap", "mask", "gloves"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/crops_plus_sdk")
ap.add_argument("--arch", default="efficientnet_b0")
ap.add_argument("--epochs", type=int, default=45)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--smooth", type=float, default=0.05)
ap.add_argument("--patience", type=int, default=12)
ap.add_argument("--out", default="runs/shared")
a = ap.parse_args()
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)

def stem_of(p):
    s = Path(p).stem
    for suf in ("__o", "__h", "__g"):
        s = s.split(suf)[0]
    return s

# ---- one row per crop FILE, carrying whatever labels that file has
def build(split):
    rows = {}
    for j, at in enumerate(ATT):
        for cls, y in (("present", 1), ("absent", 0)):
            for p in glob.glob(f"{a.data}/{at}/{split}/{cls}/*.jpg"):
                r = rows.setdefault(p, {"path": p, "y": [0.]*3, "m": [0.]*3})
                r["y"][j] = float(y); r["m"][j] = 1.0
    # the identical head crop may appear under head_cap and mask with different
    # augmentation suffixes; merge by (stem, suffix) so both labels reach one sample
    merged = {}
    for p, r in rows.items():
        key = (stem_of(p), Path(p).name.split("__")[-1])
        if key in merged:
            for j in range(3):
                if r["m"][j]:
                    merged[key]["y"][j] = r["y"][j]; merged[key]["m"][j] = 1.0
        else:
            merged[key] = r
    return list(merged.values())

class Crops(Dataset):
    def __init__(self, rows, train): self.rows, self.train = rows, train
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        im = cv2.imread(r["path"])
        im = cv2.resize(im, (a.px, a.px)) if im is not None else np.zeros((a.px,a.px,3), np.uint8)
        if self.train:
            if np.random.rand() < 0.5: im = im[:, ::-1]
            hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[...,0] = (hsv[...,0] + np.random.randint(0,180)) % 180
            hsv[...,1] = np.clip(hsv[...,1]*np.random.uniform(0.7,1.25), 0, 255)
            hsv[...,2] = np.clip(hsv[...,2]*np.random.uniform(0.7,1.3), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if np.random.rand() < 0.25:
                im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            if np.random.rand() < 0.3:
                eh, ew = np.random.randint(a.px//6, a.px//3, 2)
                ey, ex = np.random.randint(0, a.px-eh), np.random.randint(0, a.px-ew)
                im = np.ascontiguousarray(im); im[ey:ey+eh, ex:ex+ew] = np.random.randint(0,255,3)
        x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
        return (torch.from_numpy(x.transpose(2,0,1).copy()),
                torch.tensor(r["y"], dtype=torch.float32), torch.tensor(r["m"], dtype=torch.float32))

class Shared(nn.Module):
    """One trunk, three heads. Attributes share everything except the last layer."""
    def __init__(self, arch):
        super().__init__()
        b = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1")
        if hasattr(b, "classifier") and isinstance(b.classifier, nn.Sequential):
            d = b.classifier[-1].in_features; b.classifier = nn.Identity()
        elif hasattr(b, "fc"):
            d = b.fc.in_features; b.fc = nn.Identity()
        self.trunk, self.head = b, nn.Linear(d, 3)
    def forward(self, x): return self.head(self.trunk(x))

dev = "cuda"
tr, va = build("train"), build("val")
print(f"{a.arch}: {len(tr)} train crops, {len(va)} val")
for j, at in enumerate(ATT):
    n = sum(1 for r in tr if r["m"][j]); pos = sum(1 for r in tr if r["m"][j] and r["y"][j])
    both = sum(1 for r in tr if r["m"][0] and r["m"][1])
    print(f"  {at:9} labelled on {n:5d} crops ({pos} present)")
print(f"  crops carrying BOTH head_cap and mask labels: {sum(1 for r in tr if r['m'][0] and r['m'][1])}")

net = Shared(a.arch).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
pw = []
for j in range(3):
    p = sum(1 for r in tr if r["m"][j] and r["y"][j]); n = sum(1 for r in tr if r["m"][j] and not r["y"][j])
    pw.append(max(0.25, min(4.0, n/max(1,p))))
pw = torch.tensor(pw, device=dev)
dl = DataLoader(Crops(tr, True), batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
dv = DataLoader(Crops(va, False), batch_size=a.batch, num_workers=4)
out = Path(a.out) / a.arch; out.mkdir(parents=True, exist_ok=True)
best, bad, t0 = -1.0, 0, time.time()
for ep in range(1, a.epochs+1):
    net.train()
    for x, y, m in dl:
        x, y, m = x.to(dev), y.to(dev), m.to(dev)
        ys = y*(1-a.smooth) + 0.5*a.smooth
        loss = (nn.functional.binary_cross_entropy_with_logits(
            net(x), ys, pos_weight=pw, reduction="none") * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    net.eval(); S, Y, M = [], [], []
    with torch.no_grad():
        for x, y, m in dv:
            S.append(torch.sigmoid(net(x.to(dev))).cpu().numpy()); Y.append(y.numpy()); M.append(m.numpy())
    S, Y, M = np.concatenate(S), np.concatenate(Y), np.concatenate(M)
    aucs = {}
    for j, at in enumerate(ATT):
        k = M[:, j] > 0
        aucs[at] = roc_auc_score(Y[k, j], S[k, j]) if k.sum() > 5 and len(set(Y[k, j])) > 1 else float("nan")
    mean = float(np.nanmean(list(aucs.values())))
    if mean > best:
        best, bad = mean, 0
        torch.save({"model": net.state_dict(), "arch": a.arch, "attrs": ATT}, out / "shared_best.pt")
    else:
        bad += 1
    if ep % 10 == 0 or ep == 1:
        print(f"  ep {ep:3d} " + "  ".join(f"{at} {aucs[at]:.3f}" for at in ATT) + f"   mean {mean:.3f}")
    if bad >= a.patience:
        print(f"  early stop at ep {ep}"); break
print(f"{a.arch} shared: best mean val AUC {best:.4f}  ({time.time()-t0:.0f}s) -> {out/'shared_best.pt'}")
