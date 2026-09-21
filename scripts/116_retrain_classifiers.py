"""Retrain the head_cap and mask classifiers on the audited data (data/clean), every architecture.

    head_cap  data/clean/head_cap            (head crops, device geometry: head_roi + _crop)
    mask      data/clean/mask_face/g0.8      (face crops, device geometry: face_roi 0.8 + _crop)

Variants per attribute
    resnet34, efficientnet_b0, efficientnet_b1       ImageNet init, script-72 recipe
    <arch>_ft                                        fine-tuned from the currently deployed trunk
                                                     (head_cap resnet34, mask efficientnet_b0)
    yolo26n-cls, yolo26s-cls                         Ultralytics classify, COCO/ImageNet-pretrained

Model selection and the decision threshold come from VAL only (violation =
item absent is the positive class, best MCC over thresholds). COCO is the test
set and is never touched here -- scripts/117_eval_coco_ppe.py scores it.

    python scripts/116_retrain_classifiers.py [--attrs head_cap,mask] [--variants ...]
Outputs runs/retrain_clean/<attr>/<variant>/{best.pt|weights/best.pt} and runs/retrain_clean/<attr>/summary.json
"""
import argparse, glob, json, random, shutil, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--attrs", default="head_cap,mask")
ap.add_argument("--variants", default="resnet34,efficientnet_b0,efficientnet_b1,ft,yolo26n-cls,yolo26s-cls")
ap.add_argument("--epochs", type=int, default=45)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--smooth", type=float, default=0.05)
ap.add_argument("--patience", type=int, default=12)
ap.add_argument("--out", default="runs/retrain_clean")
ap.add_argument("--extra", action="store_true", help="also train/validate on data/clean/sppg_<attr> (reviewed sppg crops)")
a = ap.parse_args()
DATA = {"head_cap": ROOT/"data/clean/head_cap", "mask": ROOT/"data/clean/mask_face/g0.8"}
DEPLOYED = {"head_cap": ("resnet34", ROOT/"runs/fixed_sdk/resnet34/head_cap_best.pt"),
            "mask": ("efficientnet_b0", ROOT/"runs/mask_face/g0.8_deployed/mask_best.pt")}
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = "cuda"


def seed():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)


EXTRA = {"head_cap": ROOT/"data/clean/sppg_head_cap", "mask": ROOT/"data/clean/sppg_mask"}


def roots(attr):
    return [DATA[attr]] + ([EXTRA[attr]] if a.extra else [])


def items(attr, split):
    return [(p, 1 if c == "present" else 0) for r in roots(attr) for c in ("present", "absent")
            for p in sorted(glob.glob(str(r/split/c/"*.jpg")))]


def norm(im):
    x = (cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
    return x.transpose(2, 0, 1)


class Crops(torch.utils.data.Dataset):
    def __init__(self, its, train): self.its, self.train = its, train
    def __len__(self): return len(self.its)
    def __getitem__(self, i):
        p, y = self.its[i]
        im = cv2.resize(cv2.imread(p), (a.px, a.px))
        if self.train:
            if np.random.rand() < 0.5: im = im[:, ::-1]
            hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 1] = np.clip(hsv[..., 1]*np.random.uniform(0.7, 1.25), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2]*np.random.uniform(0.7, 1.3), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if np.random.rand() < 0.3:
                eh, ew = np.random.randint(a.px//6, a.px//3, 2)
                ey, ex = np.random.randint(0, a.px-eh), np.random.randint(0, a.px-ew)
                im = np.ascontiguousarray(im); im[ey:ey+eh, ex:ex+ew] = np.random.randint(0, 255, 3)
        return torch.from_numpy(norm(im).copy()), torch.tensor(float(y))


def best_threshold(P, Y):
    """p = P(item present). Violation (absent) is positive: flag when p < t."""
    best = (0.5, -1.0)
    for t in np.arange(0.05, 0.96, 0.025):
        q = (P < t).astype(int)
        m = matthews_corrcoef(1-Y, q) if len(set(q)) > 1 else 0.0
        if m > best[1]: best = (float(t), float(m))
    return best


def build(arch, init=None):
    net = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1")
    if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, 1)
    else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
    if init is not None:
        sd = torch.load(init, map_location="cpu", weights_only=False)
        net.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd)
    return net.to(dev)


@torch.no_grad()
def predict_torch(net, its):
    net.eval(); P = []
    for x, _ in torch.utils.data.DataLoader(Crops(its, False), batch_size=256, num_workers=6):
        P += torch.sigmoid(net(x.to(dev))[:, 0]).cpu().tolist()
    return np.array(P)


def train_torch(attr, arch, init, d):
    seed()
    tr, va = items(attr, "train"), items(attr, "val"); Yv = np.array([y for _, y in va])
    net = build(arch, init)
    pos = sum(y for _, y in tr); pw = torch.tensor(max(0.25, min(4.0, (len(tr)-pos)/max(1, pos))), device=dev)
    lr = a.lr/3 if init else a.lr
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dl = torch.utils.data.DataLoader(Crops(tr, True), batch_size=a.batch, shuffle=True, num_workers=8, drop_last=True)
    best, bad, ep = (-2.0, None, 0.5, 0), 0, 0
    for ep in range(1, a.epochs+1):
        net.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x)[:, 0], y*(1-a.smooth) + 0.5*a.smooth, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        t, m = best_threshold(predict_torch(net, va), Yv)
        if m > best[0]:
            best, bad = (m, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, t, ep), 0
        else:
            bad += 1
        if bad >= a.patience: break
    torch.save(best[1], d/"best.pt")
    return {"kind": "torch", "arch": arch, "init": str(init) if init else "imagenet", "weights": str(d/"best.pt"),
            "val_mcc": round(best[0], 4), "threshold": round(best[2], 3), "best_epoch": best[3], "epochs_run": ep,
            "train": len(tr), "val": len(va)}


def train_yolo(attr, model, d):
    from ultralytics import YOLO
    seed()
    root = d/"dataset"                                   # ultralytics wants root/{train,val}/<class>
    if root.exists(): shutil.rmtree(root)
    for sp in ("train", "val"):                          # one folder per class, files linked in from every root
        for c in ("present", "absent"):
            (root/sp/c).mkdir(parents=True)
            for r in roots(attr):
                for f in (r/sp/c).glob("*.jpg"): (root/sp/c/f"{r.name}__{f.name}").symlink_to(f)
    m = YOLO(str(ROOT/f"{model}.pt"))
    m.train(data=str(root), epochs=a.epochs, imgsz=a.px, batch=a.batch, patience=a.patience,
            project=str(d), name="train", exist_ok=True, device=0, workers=8, seed=0, plots=False,
            fliplr=0.5, hsv_s=0.3, hsv_v=0.3, erasing=0.3, verbose=False)
    w = d/"train/weights/best.pt"
    net = YOLO(str(w)); names = net.names; pi = [k for k, v in names.items() if v == "present"][0]
    va = items(attr, "val"); Yv = np.array([y for _, y in va])
    P = np.array([float(r.probs.data[pi]) for s in range(0, len(va), 256)
                  for r in net.predict([p for p, _ in va[s:s+256]], imgsz=a.px, verbose=False)])
    t, mcc = best_threshold(P, Yv)
    return {"kind": "yolo", "arch": model, "init": "pretrained", "weights": str(w), "present_index": int(pi),
            "val_mcc": round(mcc, 4), "threshold": round(t, 3), "train": len(items(attr, "train")), "val": len(va)}


for attr in a.attrs.split(","):
    out = ROOT/a.out/attr; out.mkdir(parents=True, exist_ok=True)
    sp = out/"summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    for v in a.variants.split(","):
        tag = f"{DEPLOYED[attr][0]}_ft" if v == "ft" else v
        d = out/tag; d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if v.startswith("yolo"): r = train_yolo(attr, v, d)
        elif v == "ft": r = train_torch(attr, DEPLOYED[attr][0], DEPLOYED[attr][1], d)
        else: r = train_torch(attr, v, None, d)
        r["minutes"] = round((time.time()-t0)/60, 1)
        summary[tag] = r
        sp.write_text(json.dumps(summary, indent=2))
        print(f"{attr:9} {tag:24} val MCC {r['val_mcc']:.3f} @ {r['threshold']:.2f}  ({r['minutes']} min)", flush=True)
