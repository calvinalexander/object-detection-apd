"""Mine head/hand ROIs from the 60-site pool to fix head_cap coverage.

The existing crops_cls pool is 89-95% one camera in one room: every `present`
example is a green bouffant cap, frontal, at the same desk. A blue cap in
profile at another site matches nothing in it (best cosine 0.60), so the
classifier is being asked to extrapolate rather than recognise.

Selection is farthest-point sampling in DINOv2 space, seeded with the existing
pool. That spends the labelling budget on appearances the training set does
NOT already contain, instead of buying more of the same room.
"""
import argparse, glob, json, sys
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.overlay import clean, detect
from apd.pose import build_rois
from apd.schema import Attribute, RoiGeometry, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--attribute", default=Attribute.HEAD_CAP.value)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--per-image", type=int, default=3)
ap.add_argument("--pose-conf", type=float, default=0.40)
ap.add_argument("--pose-imgsz", type=int, default=1280)
ap.add_argument("--out", default="artifacts/site_crops")
args = ap.parse_args()

part = "head" if args.attribute in (Attribute.HEAD_CAP.value, Attribute.MASK.value) else "hand"
out = Path(args.out) / args.attribute
(out / "pool").mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO
pose = YOLO(".work/yolo11x-pose.pt")

imgs = sorted(glob.glob("data/sites/*/*.jpg"))
print(f"pose over {len(imgs)} site images ...", flush=True)
records = []
for i in range(0, len(imgs), 16):
    batch = imgs[i:i + 16]
    # The dashboard's burned-in verdict is removed BEFORE anything looks at the
    # frame: it states the answer in plain text and correlates with violation by
    # construction, so neither the labeller nor a classifier may see it.
    frames = {p: clean(cv2.imread(p)) for p in batch}
    overlays = {p: detect(cv2.imread(p)) for p in batch}
    for path, r in zip(batch, pose.predict(list(frames.values()), imgsz=args.pose_imgsz,
                                           conf=args.pose_conf, verbose=False, device=0)):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        src, ov = frames[path], overlays[path]
        kp = r.keypoints.data.cpu().numpy()
        persons = build_rois(r.boxes.xyxy.cpu().numpy(), kp)
        taken = 0
        for pid, p in enumerate(persons):
            if taken >= args.per_image:
                break
            rois = [p.head] if part == "head" else list(p.hands)
            if part == "head" and (p.head is None or not p.head_localizable or p.head_collision):
                continue
            for j, box in enumerate(rois):
                if box is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                if x2 - x1 < 24 or y2 - y1 < 24:
                    continue
                if ov.covers((x1, y1, x2, y2)):        # crop would carry a caption
                    continue
                pad = int(0.35 * max(x2 - x1, y2 - y1))
                crop = src[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                if crop.size == 0:
                    continue
                site = Path(path).parent.name
                name = f"{site}__{Path(path).stem}_{pid}{j}.jpg"
                cv2.imwrite(str(out / "pool" / name), crop)
                records.append({"crop": name, "site": site, "image": path,
                                "person": pid, "box": [x1, y1, x2, y2],
                                "person_box": [int(v) for v in p.person_box]})
                taken += 1
    if (i // 16) % 20 == 0:
        print(f"  {i}/{len(imgs)}  crops={len(records)}", flush=True)
print(f"pool: {len(records)} crops from {len({r['site'] for r in records})} sites")

# --- farthest-point sampling against the EXISTING pool -------------------
dev = "cuda"
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(dev).eval()
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

def embed(paths, bs=64):
    outs = []
    for i in range(0, len(paths), bs):
        b = []
        for p in paths[i:i + bs]:
            im = cv2.imread(str(p))
            im = cv2.cvtColor(cv2.resize(im, (126, 126)), cv2.COLOR_BGR2RGB) / 255.0
            b.append(((im - MEAN) / STD).transpose(2, 0, 1))
        with torch.no_grad():
            f = dino(torch.tensor(np.array(b), dtype=torch.float32, device=dev))
        outs.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(outs)

have = sorted(glob.glob(f"data/crops_cls/{args.attribute}/*/*/*.jpg"))
cand = [out / "pool" / r["crop"] for r in records]
print(f"embedding {len(have)} existing + {len(cand)} candidates ...", flush=True)
E_have, E_cand = embed(have), embed(cand)

# distance to the nearest thing already labelled; greedily take the farthest,
# then update so we don't pick 400 copies of the same novel appearance.
d = 1.0 - (E_cand @ E_have.T).max(axis=1)
picked = []
for _ in range(min(args.n, len(cand))):
    k = int(np.argmax(d))
    picked.append(k)
    d = np.minimum(d, 1.0 - (E_cand @ E_cand[k]))
    d[k] = -1

sel = [records[k] | {"novelty": float(1.0 - (E_cand[k] @ E_have.T).max())} for k in picked]
(out / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
print(f"\nselected {len(sel)} crops from {len({r['site'] for r in sel})} sites")
print(f"novelty (1 - cos to nearest labelled crop): "
      f"min {min(r['novelty'] for r in sel):.3f}  "
      f"median {np.median([r['novelty'] for r in sel]):.3f}  "
      f"max {max(r['novelty'] for r in sel):.3f}")
base = 1.0 - (E_have @ E_have.T - 2 * np.eye(len(E_have))).max(axis=1)
print(f"for reference, existing pool's own nearest-neighbour distance: median {np.median(base):.3f}")
