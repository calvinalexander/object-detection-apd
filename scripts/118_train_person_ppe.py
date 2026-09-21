"""One person crop in, all three PPE items out: a single multi-label classifier.

    person crop (pose box + 18% pad, letterboxed to --px)  ->  backbone  ->  3 logits
                                                                            head_cap, mask, gloves
Labels come from the audited detector set data/clean/det7, where every image is
already one person with that person's PPE boxes:
    head_cap  present if a head_cap box and no no_head_cap box; absent if a no_head_cap box
    mask      same with mask / no_mask
    gloves    absent if any no_gloves box (one bare hand is a violation); present if gloves boxes only
    no box for an item -> unknown, masked out of that item's loss (never guessed)
COCO frames are not in det7, so the COCO test stays untouched.

ONE checkpoint (the point of the design: one forward pass per person) is chosen
on the mean VAL MCC of the three items; each item then gets its own decision
threshold at that checkpoint (violation = absent is positive).

    python scripts/118_train_person_ppe.py [--archs resnet34,efficientnet_b0,efficientnet_b1] [--px 320]
Writes data/clean/person_ppe/labels.csv and runs/person_ppe/<arch>_<px>/{best.pt, summary.json}
"""
import argparse, collections, csv, hashlib, json, random, time
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--archs", default="resnet34,efficientnet_b0,efficientnet_b1")
ap.add_argument("--px", type=int, default=320)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--out", default="runs/person_ppe")
ap.add_argument("--extra", default="", help="add data/label_sppg/labels.csv (reviewed sppg people), split by scene")
ap.add_argument("--tag", default="")
ap.add_argument("--rec", default="", help="add data/label_recording/labels.csv (device camera); clips split 20%% test / 15%% val / rest train")
a = ap.parse_args()
ITEMS = ["head_cap", "mask", "gloves"]
D = ROOT/"data/clean/det7"
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = "cuda"


def build_labels():
    L = ROOT/"data/clean/person_ppe"/(("labels_" + a.tag) if a.tag else "labels") 
    L = L.with_suffix(".csv")
    rows = []
    for sp in ("train", "val"):
        for lp in sorted((D/"labels"/sp).glob("*.txt")):
            cls = {int(l.split()[0]) for l in lp.read_text().splitlines() if l.strip()}
            lab = lambda pos, neg: 0 if neg in cls else 1 if pos in cls else -1
            rows.append([sp, str((D/"images"/sp/(lp.stem + ".jpg")).relative_to(ROOT)),
                         lab(2, 5), lab(3, 6), lab(1, 4)])
    if a.rec:
        V = {"present": 1, "absent": 0, "unclear": -1}
        recs = [r for r in csv.DictReader(open(ROOT/a.rec))
                if "not_person" not in (r["head_cap"], r["mask"], r["gloves"])]
        by_clip = collections.defaultdict(list)
        for r in recs: by_clip[r["clip"]].append(r)
        # TEST clips chosen so every item has both classes in the test set, which a
        # plain hash split does not give: gloves-present people sit in a few clips.
        need = {(it, c): 12 for it in ITEMS for c in ("present", "absent")}
        order = sorted(by_clip, key=lambda c: -sum(1 for r in by_clip[c] if r["gloves"] == "present"))
        test_clips, have = set(), collections.Counter()
        for c in order:
            if all(have[k] >= v for k, v in need.items()): break
            test_clips.add(c)
            for r in by_clip[c]:
                for it in ITEMS:
                    if r[it] in ("present", "absent"): have[(it, r[it])] += 1
        held = []
        for r in recs:
            sp = ("test" if r["clip"] in test_clips else
                  "val" if int(hashlib.md5(("c" + r["clip"]).encode()).hexdigest(), 16) % 100 < 18 else "train")
            row = [sp, r["person"], V[r["head_cap"]], V[r["mask"]], V[r["gloves"]]]
            (held if sp == "test" else rows).append(row)
        (ROOT/"data/label_recording/test_split.csv").write_text(
            "split,image,head_cap,mask,gloves\n" + "\n".join(",".join(map(str, r)) for r in held) + "\n")
        print(f"device camera: TEST {len(held)} people from {len(test_clips)} clips, "
              + ", ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(have.items())), flush=True)
    if a.extra:
        V = {"present": 1, "absent": 0, "unclear": -1}
        for r in csv.DictReader(open(ROOT/a.extra)):
            if "not_person" in (r["head_cap"], r["mask"], r["gloves"]): continue
            sp = "val" if int(hashlib.md5(("s" + r["scene"]).encode()).hexdigest(), 16) % 100 < 15 else "train"
            rows.append([sp, r["person"], V[r["head_cap"]], V[r["mask"]], V[r["gloves"]]])
    L.parent.mkdir(parents=True, exist_ok=True)
    with open(L, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["split", "image", *ITEMS]); w.writerows(rows)
    return rows


def letterbox(im, s):
    h, w = im.shape[:2]; r = s / max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    out = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    out[y:y+im.shape[0], x:x+im.shape[1]] = im
    return out


def prep(im, s):
    """What the device will run: BGR person crop -> letterbox -> RGB, ImageNet norm, CHW."""
    x = (cv2.cvtColor(letterbox(im, s), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
    return x.transpose(2, 0, 1)


class People(torch.utils.data.Dataset):
    def __init__(self, rows, train): self.rows, self.train = rows, train
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        _, p, *y = self.rows[i]
        im = cv2.imread(str(ROOT/p))
        if self.train:
            h, w = im.shape[:2]                                   # mild re-crop: the pose box is never exact
            dx, dy = int(w*np.random.uniform(0, .06)), int(h*np.random.uniform(0, .06))
            im = im[dy:h-int(h*np.random.uniform(0, .06)), dx:w-int(w*np.random.uniform(0, .06))]
            if np.random.rand() < 0.5: im = im[:, ::-1]
            hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 1] = np.clip(hsv[..., 1]*np.random.uniform(0.7, 1.25), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2]*np.random.uniform(0.7, 1.3), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if np.random.rand() < 0.3:                            # downscale-upscale: CCTV people are small
                f = np.random.uniform(0.35, 0.8)
                im = cv2.resize(cv2.resize(im, None, fx=f, fy=f), (im.shape[1], im.shape[0]))
        return torch.from_numpy(prep(np.ascontiguousarray(im), a.px)), torch.tensor([float(v) for v in y])


def build(arch):
    net = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1")
    if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, 3)
    else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 3)
    return net.to(dev)


def best_threshold(p, y):
    best = (0.5, -1.0)
    for t in np.arange(0.05, 0.96, 0.025):
        q = (p < t).astype(int)
        m = matthews_corrcoef(1-y, q) if len(set(q)) > 1 else 0.0
        if m > best[1]: best = (float(t), float(m))
    return best


if __name__ == "__main__":
    rows = build_labels()
    tr = [r for r in rows if r[0] == "train"]; va = [r for r in rows if r[0] == "val"]
    Yv = np.array([r[2:] for r in va])
    for arch in a.archs.split(","):
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        tag = f"{arch}_{a.px}" + (f"_{a.tag}" if a.tag else ""); out = ROOT/a.out/tag; out.mkdir(parents=True, exist_ok=True)
        Ytr = np.array([r[2:] for r in tr])
        pw = torch.tensor([max(0.25, min(4.0, ((Ytr[:, k] == 0).sum())/max(1, (Ytr[:, k] == 1).sum()))) for k in range(3)], device=dev)
        net = build(arch)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        dl = torch.utils.data.DataLoader(People(tr, True), batch_size=a.batch, shuffle=True, num_workers=8, drop_last=True)
        dv = torch.utils.data.DataLoader(People(va, False), batch_size=128, num_workers=8)
        best = (-2.0, None, 0); bad, t0 = 0, time.time()
        for ep in range(1, a.epochs+1):
            net.train()
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                known = (y >= 0).float()
                ys = y.clamp(min=0)*0.95 + 0.025                      # label smoothing
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), ys, pos_weight=pw, reduction="none")
                loss = (loss*known).sum() / known.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            net.eval(); P = []
            with torch.no_grad():
                for x, _ in dv: P.append(torch.sigmoid(net(x.to(dev))).cpu().numpy())
            P = np.concatenate(P)
            per = {it: best_threshold(P[Yv[:, k] >= 0, k], Yv[Yv[:, k] >= 0, k]) for k, it in enumerate(ITEMS)}
            mean = float(np.mean([m for _, m in per.values()]))
            if mean > best[0]:
                best, bad = (mean, per, ep), 0
                torch.save(net.state_dict(), out/"best.pt")
            else:
                bad += 1
            print(f"{tag} ep {ep:2d} mean {mean:.3f} " + " ".join(f"{it} {per[it][1]:.3f}" for it in ITEMS)
                  + f"  | best mean {best[0]:.3f} @ ep {best[2]}", flush=True)
            if bad >= a.patience: break
        summary = {"arch": arch, "px": a.px, "tag": a.tag, "data": "det7" + (" + " + a.extra if a.extra else ""), "weights": str(out/"best.pt"), "outputs": ITEMS,
                   "best_epoch": best[2], "val_mean_mcc": round(best[0], 4), "epochs_run": ep,
                   "minutes": round((time.time()-t0)/60, 1), "train": len(tr), "val": len(va),
                   "items": {it: {"val_mcc": round(best[1][it][1], 4), "threshold": round(best[1][it][0], 3),
                                  "val_known": int((Yv[:, k] >= 0).sum())} for k, it in enumerate(ITEMS)}}
        (out/"summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{tag}: " + ", ".join(f"{it} val MCC {v['val_mcc']:.3f} @ {v['threshold']:.2f}"
                                    for it, v in summary["items"].items()) + f"  (epoch {best[2]}, {summary['minutes']} min)", flush=True)
