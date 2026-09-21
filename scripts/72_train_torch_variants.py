"""Torchvision variants on the same clean split, with the overfitting addressed.

The published EfficientNet-B1 was trained ad-hoc with no held-out stopping
signal and reached a 0.0003 training loss -- memorisation. Three changes here:
an internal validation split carved by SITE (so stopping is not judged on
near-duplicates), label smoothing plus random erasing, and a smaller default
backbone, since ~3k crops does not support 6.5M parameters.
"""
import argparse, glob, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/crops_cocoheld")
ap.add_argument("--arch", default="efficientnet_b0")
ap.add_argument("--attrs", default="head_cap,mask,gloves")
ap.add_argument("--epochs", type=int, default=45)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--smooth", type=float, default=0.05)
ap.add_argument("--patience", type=int, default=12)
ap.add_argument("--out", default="runs/bakeoff_torch")
a = ap.parse_args()
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)

def norm(im):
    x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
    return x.transpose(2,0,1)

class Crops(Dataset):
    def __init__(self, items, train): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = cv2.resize(cv2.imread(p), (a.px, a.px))
        if self.train:
            if np.random.rand() < 0.5: im = im[:, ::-1]
            hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[...,0] = (hsv[...,0] + np.random.randint(0,180)) % 180
            hsv[...,1] = np.clip(hsv[...,1]*np.random.uniform(0.7,1.25), 0, 255)
            hsv[...,2] = np.clip(hsv[...,2]*np.random.uniform(0.7,1.3), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if np.random.rand() < 0.25:
                im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            if np.random.rand() < 0.3:                      # random erasing
                eh, ew = np.random.randint(a.px//6, a.px//3, 2)
                ey, ex = np.random.randint(0, a.px-eh), np.random.randint(0, a.px-ew)
                im = np.ascontiguousarray(im); im[ey:ey+eh, ex:ex+ew] = np.random.randint(0,255,3)
        return torch.from_numpy(norm(im).copy()), torch.tensor(float(y))

def build(arch):
    net = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1")
    if hasattr(net, "classifier") and isinstance(net.classifier, torch.nn.Sequential):
        net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
    elif hasattr(net, "fc"):
        net.fc = torch.nn.Linear(net.fc.in_features, 1)
    else:
        net.classifier = torch.nn.Linear(net.classifier.in_features, 1)
    return net

dev = "cuda"; out = Path(a.out) / a.arch; out.mkdir(parents=True, exist_ok=True)
for attr in a.attrs.split(","):
    def load(sp):
        return [(p, 1 if cls == "present" else 0)
                for cls in ("present", "absent")
                for p in glob.glob(f"{a.data}/{attr}/{sp}/{cls}/*.jpg")]
    tr, va = load("train"), load("val")
    if not tr or not va or len({y for _, y in va}) < 2:
        print(f"{attr}: unusable split"); continue
    pos = sum(y for _, y in tr)
    net = build(a.arch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    pw = torch.tensor([max(0.25, min(4.0, (len(tr)-pos)/max(1,pos)))], device=dev)
    dl = DataLoader(Crops(tr, True), batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    dv = DataLoader(Crops(va, False), batch_size=a.batch, num_workers=4)
    best, bad, t0 = -1.0, 0, time.time()
    for ep in range(1, a.epochs+1):
        net.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            ys = y*(1-a.smooth) + 0.5*a.smooth              # label smoothing
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x)[:,0], ys, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        sched.step()
        net.eval(); S, Y = [], []
        with torch.no_grad():
            for x, y in dv:
                S.append(torch.sigmoid(net(x.to(dev))[:,0]).cpu().numpy()); Y.append(y.numpy())
        S, Y = np.concatenate(S), np.concatenate(Y)
        auc = roc_auc_score(Y, S)
        if auc > best:
            best, bad = auc, 0
            torch.save(net.state_dict(), out / f"{attr}_best.pt")
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  {a.arch} {attr} ep {ep:3d} loss {tot/len(dl):.4f} valAUC {auc:.4f} best {best:.4f}")
        if bad >= a.patience:
            print(f"  early stop at ep {ep} (no gain in {a.patience})"); break
    print(f"{a.arch} {attr}: best internal-val AUC {best:.4f}  ({time.time()-t0:.0f}s)")
