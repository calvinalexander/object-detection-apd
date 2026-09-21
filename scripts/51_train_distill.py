"""One multi-head classifier over the person crop -- the deployable stage 2.

Three attributes share a backbone because they share a subject: the same person,
the same lighting, the same camera. Three separate networks pay for that context
three times, which is exactly the cost the edge cannot afford.

`unclear` is masked out of the loss rather than treated as a class. An abstention
is an absence of evidence, not a third appearance to learn; training a head to
predict it would teach the model to reproduce our gating rules instead of the
equipment. At inference the model can still abstain, via the confidence margin.

Colour handling follows what was measured: hue jitter is aggressive (the blue-cap
failure was a hue shortcut) but saturation is preserved, since a glove or a mask
is partly told from skin by colour contrast.
"""
import argparse, collections, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, assert_contract

assert_contract()
ATTRS = [a.value for a in Attribute]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/distill")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--size", default="192x96")
ap.add_argument("--out", default="runs/distill")
a = ap.parse_args()
H, W = (int(v) for v in a.size.split("x"))
root = Path(a.data)
rows = [json.loads(l) for l in (root / "index.jsonl").read_text().splitlines()]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Crops(Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = cv2.imread(str(root / "crops" / r["crop"]))
        im = cv2.resize(im, (W, H))
        if self.train:
            if np.random.rand() < 0.5:
                im = im[:, ::-1]
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + np.random.randint(0, 180)) % 180   # any hue
            hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.8, 1.2), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.7, 1.3), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        x = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        y = np.zeros(len(ATTRS), np.float32)
        m = np.zeros(len(ATTRS), np.float32)
        for j, at in enumerate(ATTRS):
            v = r["labels"][at]
            if v != "unclear":
                y[j] = float(v == "present"); m[j] = 1.0
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(y), torch.from_numpy(m)


class MultiHead(nn.Module):
    def __init__(self, n):
        super().__init__()
        b = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(*list(b.children())[:-1])
        self.head = nn.Linear(512, n)

    def forward(self, x):
        return self.head(self.stem(x).flatten(1))


dev = "cuda"
tr = [r for r in rows if r["split"] == "train"]
va = [r for r in rows if r["split"] == "val"]
dl_tr = DataLoader(Crops(tr, True), batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
dl_va = DataLoader(Crops(va, False), batch_size=a.batch, num_workers=4)

# Class imbalance is severe for gloves (184 present vs 744 absent in train), so
# each head gets its own positive weight rather than one global one.
pw = []
for at in ATTRS:
    p = sum(1 for r in tr if r["labels"][at] == "present")
    n = sum(1 for r in tr if r["labels"][at] == "absent")
    pw.append(max(0.2, min(5.0, n / max(1, p))))
print("pos_weight per head:", {at: round(w, 2) for at, w in zip(ATTRS, pw)})

net = MultiHead(len(ATTRS)).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
pw_t = torch.tensor(pw, device=dev)
best, out = -1.0, Path(a.out); out.mkdir(parents=True, exist_ok=True)
from sklearn.metrics import roc_auc_score

for ep in range(1, a.epochs + 1):
    net.train(); tot = 0.0
    for x, y, m in dl_tr:
        x, y, m = x.to(dev), y.to(dev), m.to(dev)
        loss = (nn.functional.binary_cross_entropy_with_logits(
            net(x), y, pos_weight=pw_t, reduction="none") * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss)
    sched.step()
    net.eval(); S, Y, M = [], [], []
    with torch.no_grad():
        for x, y, m in dl_va:
            S.append(torch.sigmoid(net(x.to(dev))).cpu().numpy()); Y.append(y.numpy()); M.append(m.numpy())
    S, Y, M = np.concatenate(S), np.concatenate(Y), np.concatenate(M)
    aucs = {}
    for j, at in enumerate(ATTRS):
        k = M[:, j] > 0
        aucs[at] = roc_auc_score(Y[k, j], S[k, j]) if len(set(Y[k, j])) > 1 else float("nan")
    mean = float(np.nanmean(list(aucs.values())))
    if mean > best:
        best = mean
        torch.save({"model": net.state_dict(), "attrs": ATTRS, "size": [H, W]}, out / "best.pt")
    if ep % 5 == 0 or ep == 1:
        print(f"  ep {ep:3d}  loss {tot/len(dl_tr):.4f}  " +
              "  ".join(f"{at} {aucs[at]:.3f}" for at in ATTRS) + f"   mean {mean:.3f}")
print(f"\nbest mean AUC {best:.3f} -> {out/'best.pt'}")
