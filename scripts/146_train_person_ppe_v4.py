"""Person PPE model v4: v3's data with the one-by-one audit applied, and the
"unclear" rows kept from swamping the presence heads.

Two changes over scripts/127 (which produced v2 and v3):

1. AUDIT. data/clean/audit_v3_overrides.csv, written by scripts/145 from the
   per-person audit, is applied to the det7 rows: 45 labels flipped (the item was
   plainly there), 69 turned into "unclear" (the item is not in the crop, or the
   crop holds two people and the label fits the other one), 11 rule fixes (navy
   beret counts as head_cap, cloth masks do not count as mask, beanies do not
   count as head_cap).

2. SAMPLING. Most rows carry no presence label at all for mask (70% unclear) and
   gloves (75%), so a uniform epoch spends most of its gradient on the
   judgeability head and starves the presence heads. A WeightedRandomSampler
   caps the share of presence-free rows per epoch at --unclear-share, keeping
   every row reachable rather than dropping any.

    python scripts/146_train_person_ppe_v4.py [--unclear-share 0.35]
Writes runs/person_ppe_v4/<arch>_<px>/{best.pt, summary.json}
"""
import argparse, collections, csv, hashlib, json, random, time
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--archs", default="efficientnet_b0")
ap.add_argument("--px", type=int, default=320)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--out", default="runs/person_ppe_v4")
ap.add_argument("--unclear-share", type=float, default=0.35,
                help="max share of an epoch spent on rows with no presence label")
ap.add_argument("--overrides", default="data/clean/audit_v3_overrides.csv")
ap.add_argument("--judge-weight", type=float, default=0.5,
                help="weight of the judgeability term; 1.0 lets it match the presence term")
ap.add_argument("--coco", default="data/clean/person_coco/labels.csv",
                help="COCO people, now training data (manual evaluation on the recordings instead)")
a = ap.parse_args()
ITEMS = ["head_cap", "mask", "gloves"]
D = ROOT/"data/clean/det7"
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
dev = "cuda"


def rows():
    """(split, image, [presence x3], [known x3]); -1 means 'not supervised'."""
    out = []
    for sp in ("train", "val"):
        for lp in sorted((D/"labels"/sp).glob("*.txt")):
            cls = {int(l.split()[0]) for l in lp.read_text().splitlines() if l.strip()}
            pres, known = [], []
            for pos, neg in ((2, 5), (3, 6), (1, 4)):
                if neg in cls: pres.append(0); known.append(1)
                elif pos in cls: pres.append(1); known.append(1)
                else: pres.append(-1); known.append(-1)      # no box: says nothing about visibility
            out.append([sp, str((D/"images"/sp/(lp.stem + ".jpg")).relative_to(ROOT)), pres, known])
    # head crops the audit judged unjudgeable -> judgeability supervision for det7 people
    unclear = collections.defaultdict(set)
    uf = ROOT/"data/clean/det7_unclear.csv"
    if uf.exists():
        for r in csv.DictReader(open(uf)): unclear[r["image"]].add(r["item"])
    for row in out:
        st = Path(row[1]).stem
        for it in unclear.get(st, ()):
            k = ITEMS.index(it); row[2][k] = -1; row[3][k] = 0

    # ---- the one-by-one audit (scripts/144/145)
    ov = collections.defaultdict(dict)
    of = ROOT/a.overrides
    if of.exists():
        for r in csv.DictReader(open(of)): ov[r["id"]][r["item"]] = r["new"]
    nov = collections.Counter()
    for row in out:
        for it, new in ov.get(Path(row[1]).stem, {}).items():
            k = ITEMS.index(it)
            if new == "present": row[2][k], row[3][k] = 1, 1
            elif new == "absent": row[2][k], row[3][k] = 0, 1
            else: row[2][k], row[3][k] = -1, 0          # unclear: judgeability negative
            nov[new] += 1
    if nov: print("audit overrides applied:", dict(nov))

    V = {"present": (1, 1), "absent": (0, 1), "unclear": (-1, 0), "": (-1, -1)}
    rep = set(json.loads((ROOT/"data/clean/sppg_repeats.json").read_text())) if (ROOT/"data/clean/sppg_repeats.json").exists() else set()
    for r in csv.DictReader(open(ROOT/"data/label_sppg/labels.csv")):
        if "not_person" in (r["head_cap"], r["mask"], r["gloves"]) or r["n"] in rep: continue
        sp = "val" if int(hashlib.md5(("s"+r["scene"]).encode()).hexdigest(), 16) % 100 < 15 else "train"
        pres, known = zip(*[V[r[i]] for i in ITEMS])
        out.append([sp, r["person"], list(pres), list(known)])
    cf = ROOT/a.coco
    if cf.exists():                              # COCO people: training data from now on
        for r in csv.DictReader(open(cf)):
            pres, known = zip(*[V[r[i]] for i in ITEMS])
            out.append(["train", r["person"], list(pres), list(known)])
    return out


def letterbox(im, s):
    h, w = im.shape[:2]; r = s/max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    out = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    out[y:y+im.shape[0], x:x+im.shape[1]] = im
    return out


class People(torch.utils.data.Dataset):
    def __init__(self, rs, train): self.rs, self.train = rs, train
    def __len__(self): return len(self.rs)
    def __getitem__(self, i):
        _, p, pres, known = self.rs[i]
        im = cv2.imread(str(ROOT/p))
        if self.train:
            h, w = im.shape[:2]
            im = im[int(h*np.random.uniform(0, .06)):h-int(h*np.random.uniform(0, .06)),
                    int(w*np.random.uniform(0, .06)):w-int(w*np.random.uniform(0, .06))]
            if np.random.rand() < 0.5: im = im[:, ::-1]
            hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 1] = np.clip(hsv[..., 1]*np.random.uniform(0.7, 1.25), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2]*np.random.uniform(0.7, 1.3), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if np.random.rand() < 0.3:
                f = np.random.uniform(0.35, 0.8)
                im = cv2.resize(cv2.resize(im, None, fx=f, fy=f), (im.shape[1], im.shape[0]))
        x = (cv2.cvtColor(letterbox(np.ascontiguousarray(im), a.px), cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.tensor(pres, dtype=torch.float), torch.tensor(known, dtype=torch.float)


def build(arch):
    net = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1")
    if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, 6)
    else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 6)
    return net.to(dev)                                  # [:3] presence, [3:] judgeability


def best_cut(p, y, invert=False):
    """invert=False: flag when p < t (violation). invert=True: positive when p >= t."""
    best = (0.5, -1.0)
    for t in np.arange(0.05, 0.96, 0.025):
        q = (p >= t).astype(int) if invert else (p < t).astype(int)
        m = matthews_corrcoef(y, q) if len(set(q)) > 1 else 0.0
        if m > best[1]: best = (float(t), float(m))
    return best


if __name__ == "__main__":
    R = rows()
    tr = [r for r in R if r[0] == "train"]; va = [r for r in R if r[0] == "val"]
    Pv = np.array([r[2] for r in va]); Kv = np.array([r[3] for r in va])
    print(f"train {len(tr)}  val {len(va)}")
    for k, it in enumerate(ITEMS):
        print(f"  {it:9} presence known {(Pv[:,k]>=0).sum():4d} | judgeability supervised {(Kv[:,k]>=0).sum():4d} "
              f"(unclear {(Kv[:,k]==0).sum()})")
    for arch in a.archs.split(","):
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        tag = f"{arch}_{a.px}"; out = ROOT/a.out/tag; out.mkdir(parents=True, exist_ok=True)
        Ptr = np.array([r[2] for r in tr])
        pw = torch.tensor([max(0.25, min(4.0, (Ptr[:, k] == 0).sum()/max(1, (Ptr[:, k] == 1).sum()))) for k in range(3)], device=dev)
        net = build(arch)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        # cap how much of an epoch is spent on rows that carry no presence label at all
        has_pres = np.array([any(p >= 0 for p in r[2]) for r in tr])
        npos, nneg = int(has_pres.sum()), int((~has_pres).sum())
        if nneg and 0 < a.unclear_share < 1:
            w_neg = (a.unclear_share/(1-a.unclear_share))*npos/nneg
            weights = np.where(has_pres, 1.0, w_neg)
            print(f"sampler: {npos} rows with a presence label, {nneg} without "
                  f"(weight {w_neg:.3f}, target share {a.unclear_share:.0%})")
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double), num_samples=len(tr), replacement=True)
            dl = torch.utils.data.DataLoader(People(tr, True), batch_size=a.batch, sampler=sampler,
                                             num_workers=8, drop_last=True)
        else:
            dl = torch.utils.data.DataLoader(People(tr, True), batch_size=a.batch, shuffle=True,
                                             num_workers=8, drop_last=True)
        dv = torch.utils.data.DataLoader(People(va, False), batch_size=128, num_workers=8)
        best, bad, ep = (-9, None, 0, None), 0, 0
        for ep in range(1, a.epochs+1):
            net.train()
            for x, yp, yk in dl:
                x, yp, yk = x.to(dev), yp.to(dev), yk.to(dev)
                o = net(x)
                mp, mk = (yp >= 0).float(), (yk >= 0).float()
                lp = torch.nn.functional.binary_cross_entropy_with_logits(
                    o[:, :3], yp.clamp(min=0)*0.95+0.025, pos_weight=pw, reduction="none")
                lk = torch.nn.functional.binary_cross_entropy_with_logits(
                    o[:, 3:], yk.clamp(min=0)*0.95+0.025, reduction="none")
                # per item, then across items: head_cap has twice the presence labels of
                # mask and gloves, and judgeability has more supervision than either, so a
                # pooled mean lets the plentiful signal drown the scarce one.
                lp_i = (lp*mp).sum(0)/mp.sum(0).clamp(min=1)
                lk_i = (lk*mk).sum(0)/mk.sum(0).clamp(min=1)
                loss = lp_i.mean() + a.judge_weight*lk_i.mean()
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            net.eval(); O = []
            with torch.no_grad():
                for x, _, _ in dv: O.append(torch.sigmoid(net(x.to(dev))).cpu().numpy())
            O = np.concatenate(O)
            cuts, score = {}, []
            for k, it in enumerate(ITEMS):
                mp = Pv[:, k] >= 0; tp, mcp = best_cut(O[mp, k], (Pv[mp, k] == 0).astype(int))
                mk = Kv[:, k] >= 0; tk, mck = best_cut(O[mk, 3+k], Kv[mk, k].astype(int), invert=True)
                cuts[it] = {"present_below": tp, "known_above": tk, "val_mcc_presence": round(mcp, 4),
                            "val_mcc_judgeable": round(mck, 4)}
                score += [mcp, mck]
            m = float(np.mean(score))
            if m > best[0]:
                best = (m, {kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()}, ep, cuts); bad = 0
                torch.save(best[1], out/"best.pt")
            else: bad += 1
            print(f"{tag} ep {ep:2d} mean {m:.3f} | " + " ".join(
                f"{it} p{cuts[it]['val_mcc_presence']:.2f}/k{cuts[it]['val_mcc_judgeable']:.2f}" for it in ITEMS)
                + f"  | best {best[0]:.3f} @ ep {best[2]}", flush=True)
            if bad >= a.patience: break
        (out/"summary.json").write_text(json.dumps(
            {"arch": arch, "px": a.px, "outputs": [f"present_{i}" for i in ITEMS] + [f"known_{i}" for i in ITEMS],
             "weights": str(out/"best.pt"), "best_epoch": best[2], "val_mean": round(best[0], 4),
             "cuts": best[3], "train": len(tr), "val": len(va),
             "rule": "alert when p_known >= known_above AND p_present < present_below",
             "audit_overrides": str(ROOT/a.overrides), "unclear_share": a.unclear_share,
             "judge_weight": a.judge_weight, "loss": "per-item mean, presence + judge_weight*judgeability"}, indent=2))
        print(f"{tag}: {json.dumps(best[3])}", flush=True)
