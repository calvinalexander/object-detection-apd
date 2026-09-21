"""Hold out whole sites, train on everything else, score both models on them.

Requested setup: keep the apd_v3 val crops in training (they are legitimate
training data -- they were only a problem when the same images were also the
benchmark) and instead hold out entire SITES, which neither model can memorise
its way around.

Truth for the held-out sites is the teacher's label (Opus, three views,
unanimity required). That is not human ground truth, so what this measures is
agreement with the labelling stack, not correctness in the absolute. It is still
the only label available for site frames, and it is identical for both models,
which is what makes the comparison fair.
"""
import argparse, collections, glob, json, re, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/crops_ci_inpaint")
ap.add_argument("--n-sites", type=int, default=8)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--out", default="runs/effnet_siteholdout")
a = ap.parse_args()
ATTRS = ["head_cap", "mask", "gloves"]

def stem_of(p):
    s = Path(p).stem
    for suf in ("__o", "__h", "__g"):
        s = s.split(suf)[0]
    return s

# ---- rank sites by how much labelled material they carry
per = collections.Counter()
for attr in ATTRS:
    seen = set()
    for sp in ("train", "val"):
        for cls in ("present", "absent"):
            for p in glob.glob(f"{a.data}/{attr}/{sp}/{cls}/*.jpg"):
                s = stem_of(p)
                if "__" in s: seen.add((attr, s))
    for _, s in seen: per[s.split("__")[0]] += 1
ranked = [s for s, _ in per.most_common()]
HOLD = set(ranked[:a.n_sites])
TOP2 = set(ranked[:2])
print(f"held-out sites ({len(HOLD)}): {sorted(HOLD)}")
print(f"  of which the top two: {sorted(TOP2)}\n")

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

class Crops(Dataset):
    def __init__(self, items, train): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = cv2.resize(cv2.imread(p), (a.px, a.px))
        if self.train:
            if np.random.rand() < 0.5: im = im[:, ::-1]
            hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + np.random.randint(0, 180)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.7, 1.25), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.7, 1.3), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if np.random.rand() < 0.25:
                im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.tensor(float(y))

dev = "cuda"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
manifest = {"held_out_sites": sorted(HOLD), "top2": sorted(TOP2), "attrs": {}}
for attr in ATTRS:
    train_items, hold_items = [], []
    for sp in ("train", "val"):
        for cls, y in (("present", 1), ("absent", 0)):
            for p in glob.glob(f"{a.data}/{attr}/{sp}/{cls}/*.jpg"):
                s = stem_of(p)
                site = s.split("__")[0] if "__" in s else None
                (hold_items if site in HOLD else train_items).append((p, y, s, site))
    # held-out evaluation uses originals only; augmented copies would repeat crops
    hold_items = [(p, y, s, site) for p, y, s, site in hold_items if p.endswith("__o.jpg") or "__" not in Path(p).stem]
    tr = [(p, y) for p, y, _, _ in train_items]
    pos = sum(y for _, y in tr)
    print(f"{attr}: train {len(tr)} ({pos}+/{len(tr)-pos}-)   held-out crops {len(hold_items)} "
          f"({sum(1 for _,y,_,_ in hold_items if y)}+/{sum(1 for _,y,_,_ in hold_items if not y)}-)")
    if not tr or len(hold_items) < 6:
        print("   too little to train or score"); continue

    net = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 1)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    pw = torch.tensor([max(0.25, min(4.0, (len(tr)-pos)/max(1, pos)))], device=dev)
    dl = DataLoader(Crops(tr, True), batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    ho = DataLoader(Crops([(p, y) for p, y, _, _ in hold_items], False), batch_size=a.batch, num_workers=4)
    best, t0 = -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x)[:, 0], y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 0 or ep == a.epochs:
            net.eval(); S, Y = [], []
            with torch.no_grad():
                for x, y in ho:
                    S.append(torch.sigmoid(net(x.to(dev))[:, 0]).cpu().numpy()); Y.append(y.numpy())
            S, Y = np.concatenate(S), np.concatenate(Y)
            auc = roc_auc_score(Y, S) if len(set(Y)) > 1 else float("nan")
            print(f"  ep {ep:3d}  held-out AUC {auc:.4f}  ({time.time()-t0:.0f}s)")
    torch.save(net.state_dict(), out / f"{attr}_last.pt")     # last epoch: no selection on the holdout
    net.eval(); S, Y = [], []
    with torch.no_grad():
        for x, y in ho:
            S.append(torch.sigmoid(net(x.to(dev))[:, 0]).cpu().numpy()); Y.append(y.numpy())
    S, Y = np.concatenate(S), np.concatenate(Y)
    manifest["attrs"][attr] = {
        "train_n": len(tr), "hold_n": len(hold_items),
        "preds": [{"stem": s, "site": site, "y": int(y), "p": float(pp)}
                  for (p_, y, s, site), pp in zip(hold_items, S)],
    }
(out / "holdout.json").write_text(json.dumps(manifest, indent=2))
print(f"\nwrote {out/'holdout.json'}")
