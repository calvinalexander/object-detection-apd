"""Train mask trunks on data/mask_face, one per crop geometry, same recipe as script 72.

efficientnet_b0, 128 px, ImageNet normalisation, BCE with label smoothing,
AdamW + cosine. Model selection and the decision threshold both come from VAL;
TEST is not touched here.
"""
import argparse, glob, json, random, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--geoms", default="g0.5,g0.8,g1.1,ghead")
ap.add_argument("--inits", default="imagenet,deployed")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--smooth", type=float, default=0.05)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--out", default="runs/mask_face")
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0); random.seed(0)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = "cuda"


def norm(im):
    x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
    return x.transpose(2, 0, 1)


class Crops(torch.utils.data.Dataset):
    def __init__(self, items, train): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = cv2.resize(cv2.imread(p), (a.px, a.px))
        if self.train:
            if np.random.rand() < 0.5: im = im[:, ::-1]
            hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + np.random.randint(-12, 13)) % 180     # masks come in colours
            hsv[..., 1] = np.clip(hsv[..., 1]*np.random.uniform(0.7, 1.25), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2]*np.random.uniform(0.65, 1.35), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if np.random.rand() < 0.35:                  # small faces: simulate low resolution
                s = np.random.randint(24, 72)
                im = cv2.resize(cv2.resize(im, (s, s)), (a.px, a.px))
            if np.random.rand() < 0.2:
                im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return torch.from_numpy(norm(im).copy()), torch.tensor(float(y))


def items(geom, split):
    out = []
    for cls, y in (("present", 1), ("absent", 0)):
        out += [(p, y) for p in glob.glob(str(ROOT/f"data/mask_face/{geom}/{split}/{cls}/*.jpg"))]
    return out


def build(init):
    net = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
    net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
    if init == "deployed":
        sd = torch.load(ROOT/"runs/bakeoff_torch/efficientnet_b0/mask_best.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd)
    return net.to(dev)


@torch.no_grad()
def predict(net, its):
    net.eval(); P = []; Y = []
    for x, y in torch.utils.data.DataLoader(Crops(its, False), batch_size=256, num_workers=6):
        P += torch.sigmoid(net(x.to(dev))[:, 0]).cpu().tolist(); Y += y.tolist()
    return np.array(P), np.array(Y)


def best_mcc(P, Y):
    """Violation (no mask) is positive: flag when p(mask) < t."""
    best = (0.5, -1.0)
    for t in np.arange(0.1, 0.95, 0.025):
        q = (P < t).astype(int); m = matthews_corrcoef(1-Y, q) if len(set(q)) > 1 else 0.0
        if m > best[1]: best = (float(t), float(m))
    return best


summary = {}
for geom in a.geoms.split(","):
    tr, va = items(geom, "train"), items(geom, "val")
    for init in a.inits.split(","):
        tag = f"{geom}_{init}"
        net = build(init)
        pos = sum(y for _, y in tr); pw = torch.tensor((len(tr)-pos)/max(1, pos), device=dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        dl = torch.utils.data.DataLoader(Crops(tr, True), batch_size=a.batch, shuffle=True, num_workers=8, drop_last=True)
        best = (-1, None, 0.5); bad = 0
        for ep in range(1, a.epochs+1):
            net.train()
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                ys = y*(1-a.smooth) + 0.5*a.smooth
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x)[:, 0], ys, pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            P, Y = predict(net, va); t, m = best_mcc(P, Y)
            if m > best[0]:
                best = (m, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, t); bad = 0
            else:
                bad += 1
            if bad >= a.patience: break
        d = ROOT/a.out/tag; d.mkdir(parents=True, exist_ok=True)
        torch.save(best[1], d/"mask_best.pt")
        summary[tag] = {"val_mcc": round(best[0], 4), "val_threshold": round(best[2], 3),
                        "train": len(tr), "val": len(va), "epochs": ep}
        print(f"{tag:22} val MCC {best[0]:.3f} at threshold {best[2]:.2f}  ({len(tr)} train crops, stopped at epoch {ep})", flush=True)
(ROOT/a.out/"summary.json").write_text(json.dumps(summary, indent=2))
