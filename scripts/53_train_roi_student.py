"""The deployable student: one shared backbone over the anatomical ROIs.

Replaces DINOv2-base (86M params) plus three separate YOLO26 classifiers with a
single small backbone applied to the head crop and to each hand crop. Head
features drive the head_cap and mask heads; hand features drive the glove head,
and the two hands are combined with a minimum, because the rule is that BOTH
hands must be gloved -- so the person's verdict is the weaker hand's.

`unclear` is masked out of the loss: an abstention records that we could not
see, which is a fact about the frame, not an appearance to learn.
"""
import argparse, collections, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, assert_contract

assert_contract()
HEAD_ATTRS = ["head_cap", "mask"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/distill_roi")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--arch", default="resnet18")
ap.add_argument("--out", default="runs/student")
ap.add_argument("--targets", choices=["hard", "soft"], default="hard",
                help="hard = the teacher's argmax verdict (discards its confidence and "
                     "every abstention); soft = its probabilities, which is what "
                     "distillation is actually for")
ap.add_argument("--temp", type=float, default=1.0)
a = ap.parse_args()
root = Path(a.data)
rows = [json.loads(l) for l in (root / "index.jsonl").read_text().splitlines()]
if a.targets == "soft":
    soft = {json.loads(l)["stem"]: json.loads(l)["soft"]
            for l in (root / "soft.jsonl").read_text().splitlines()}
    kept = 0
    for r in rows:
        r["soft"] = soft.get(r["stem"], {})
        kept += bool(r["soft"])
    print(f"soft targets attached to {kept}/{len(rows)} records "
          f"(hard training used only the {sum(1 for r in rows if any(v != 'unclear' for v in r['labels'].values()))} "
          f"with a decided verdict)")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(p, train):
    im = cv2.imread(str(p))
    if im is None: return np.zeros((a.px, a.px, 3), np.uint8)
    im = cv2.resize(im, (a.px, a.px))
    if train:
        if np.random.rand() < 0.5: im = im[:, ::-1]
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + np.random.randint(0, 180)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.8, 1.2), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.7, 1.3), 0, 255)
        im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return im


def norm(im):
    x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class Rois(Dataset):
    def __init__(self, rows, train): self.rows, self.train = rows, train
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        head = norm(load(root / "head" / r["head"], self.train)) if r["head"] else torch.zeros(3, a.px, a.px)
        hv = 1.0 if r["head"] else 0.0
        hands = [norm(load(root / "hand" / h, self.train)) for h in r["hands"][:2]]
        gv = 1.0 if len(hands) == 2 else 0.0
        while len(hands) < 2: hands.append(torch.zeros(3, a.px, a.px))
        y, m = np.zeros(3, np.float32), np.zeros(3, np.float32)
        for j, at in enumerate(HEAD_ATTRS + ["gloves"]):
            avail = hv if at in HEAD_ATTRS else gv
            # Val is always scored against hard labels: the student is being
            # measured on whether it is right, not on how closely it mimics.
            if a.targets == "soft" and self.train:
                sv = r.get("soft", {}).get(at)
                if sv is not None and avail:
                    # Mean of the two voters: they disagree by .17-.32 on average,
                    # and that spread is itself the signal an argmax destroys.
                    y[j] = 0.5 * (sv["v3"] + sv["v4"]); m[j] = 1.0
            else:
                v = r["labels"][at]
                if v != "unclear":
                    y[j] = float(v == "present"); m[j] = avail
        return head, hands[0], hands[1], torch.from_numpy(y), torch.from_numpy(m)


class Student(nn.Module):
    def __init__(self, arch):
        super().__init__()
        fn = getattr(torchvision.models, arch)
        b = fn(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(*list(b.children())[:-1])
        d = b.fc.in_features if hasattr(b, "fc") else 1280
        self.head_out = nn.Linear(d, 2)     # head_cap, mask
        self.hand_out = nn.Linear(d, 1)     # gloves, per hand

    def forward(self, head, h0, h1):
        f = self.stem(torch.cat([head, h0, h1], 0)).flatten(1)
        n = head.shape[0]
        hf, a0, a1 = f[:n], f[n:2 * n], f[2 * n:]
        ho = self.head_out(hf)
        g = torch.minimum(self.hand_out(a0), self.hand_out(a1))   # both hands required
        return torch.cat([ho, g], 1)


dev = "cuda"
tr = [r for r in rows if r["split"] == "train"]; va = [r for r in rows if r["split"] == "val"]
dl_tr = DataLoader(Rois(tr, True), batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
dl_va = DataLoader(Rois(va, False), batch_size=a.batch, num_workers=4)
ATT = HEAD_ATTRS + ["gloves"]
pw = []
for at in ATT:
    p = sum(1 for r in tr if r["labels"][at] == "present")
    n = sum(1 for r in tr if r["labels"][at] == "absent")
    pw.append(max(0.2, min(6.0, n / max(1, p))))
if a.targets == "soft":
    pw = [1.0] * len(ATT)   # a soft target already carries the prior; reweighting it distorts it
print(f"{a.arch} @ {a.px}px   pos_weight " + str({k: round(v, 2) for k, v in zip(ATT, pw)}))

net = Student(a.arch).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
pwt = torch.tensor(pw, device=dev)
best = -1.0; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for ep in range(1, a.epochs + 1):
    net.train(); tot = 0.0
    for head, h0, h1, y, m in dl_tr:
        head, h0, h1, y, m = (t.to(dev) for t in (head, h0, h1, y, m))
        loss = (nn.functional.binary_cross_entropy_with_logits(
            net(head, h0, h1), y, pos_weight=pwt, reduction="none") * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    sched.step()
    net.eval(); S, Y, M = [], [], []
    with torch.no_grad():
        for head, h0, h1, y, m in dl_va:
            S.append(torch.sigmoid(net(head.to(dev), h0.to(dev), h1.to(dev))).cpu().numpy())
            Y.append(y.numpy()); M.append(m.numpy())
    S, Y, M = np.concatenate(S), np.concatenate(Y), np.concatenate(M)
    aucs = {}
    for j, at in enumerate(ATT):
        k = M[:, j] > 0
        aucs[at] = roc_auc_score(Y[k, j], S[k, j]) if k.sum() and len(set(Y[k, j])) > 1 else float("nan")
    mean = float(np.nanmean(list(aucs.values())))
    if mean > best:
        best = mean
        torch.save({"model": net.state_dict(), "arch": a.arch, "px": a.px, "attrs": ATT}, out / "best.pt")
    if ep % 10 == 0 or ep == 1:
        print(f"  ep {ep:3d}  loss {tot/len(dl_tr):.4f}  " +
              "  ".join(f"{at} {aucs[at]:.3f}" for at in ATT) + f"   mean {mean:.3f}")
print(f"\nbest mean AUC {best:.3f} -> {out/'best.pt'}")
