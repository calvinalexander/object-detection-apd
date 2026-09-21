"""ROI crops for the student, keyed to the consensus verdicts.

The whole-person student stalls at .865/.660/.716 against a teacher at
.944/.902/.742. It is not capacity: at 384x192 the head occupies about 40x40
pixels, while the teacher classifies a 128x128 crop of that same head. The
resolution has to be spent where the equipment is, so the student keeps the
anatomical ROIs and saves its cost elsewhere -- one shared backbone in place of
DINOv2-base plus three separate classifiers.
"""
import argparse, collections, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.overlay import clean
from apd.pose import build_rois
from apd.schema import GEOMETRY, Attribute, assert_contract

assert_contract()
ATTRS = [a.value for a in Attribute]
ap = argparse.ArgumentParser()
ap.add_argument("--consensus", default="artifacts/consensus_ci3b.jsonl")
ap.add_argument("--out", default="data/distill_roi")
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()

from ultralytics import YOLO
pose = YOLO(".work/yolo11x-pose.pt")
recs = [json.loads(l) for l in Path(a.consensus).read_text().splitlines()]
by_frame = {(r["site"], r["id"]): r for r in recs}
import random
sites = sorted({r["site"] for r in recs})
rng = random.Random(a.seed); rng.shuffle(sites)
val_sites = set(sites[:max(1, int(len(sites) * a.val_frac))])

out = Path(a.out)
for d in ("head", "hand"):
    (out / d).mkdir(parents=True, exist_ok=True)

def crop_np(img, box, size):
    x1, y1, x2, y2 = box; pad = 0.35 * max(x2 - x1, y2 - y1)
    c = img[max(0, int(y1 - pad)):int(y2 + pad), max(0, int(x1 - pad)):int(x2 + pad)]
    return None if c.size == 0 or min(c.shape[:2]) < 8 else cv2.resize(c, (size, size))

rows, skip = [], collections.Counter()
items = list(by_frame.items())
for n, ((site, fid), r) in enumerate(items, 1):
    src = Path("data/sites") / site / f"{fid}.jpg"
    if not src.exists():
        skip["missing"] += 1; continue
    img = clean(cv2.imread(str(src)))
    rr = pose.predict(img, imgsz=1280, conf=0.40, verbose=False, device=0)[0]
    if rr.keypoints is None or not len(rr.boxes):
        skip["no person"] += 1; continue
    kp = rr.keypoints.data.cpu().numpy()
    people = build_rois(rr.boxes.xyxy.cpu().numpy(), kp)
    # The consensus recorded persons in this same order, from the same detector
    # and settings, so index alignment holds; a length mismatch means it does
    # not, and the frame is dropped rather than mislabelled.
    if len(people) != len(r["persons"]):
        skip["person count changed"] += 1; continue
    for pp, prec in zip(people, r["persons"]):
        labels = {k: prec["attrs"][k]["decision"] for k in ATTRS}
        if all(v == "unclear" for v in labels.values()):
            skip["nothing decided"] += 1; continue
        stem = f"{site}__{fid}_{prec['person']}"
        hn = None
        if pp.head is not None and not pp.head_collision:
            hc = crop_np(img, pp.head, GEOMETRY.head_crop_px)
            if hc is not None:
                hn = f"{stem}.jpg"; cv2.imwrite(str(out / "head" / hn), hc)
        hands = []
        for j, hb in enumerate(pp.hands[:2]):
            hcp = crop_np(img, hb, GEOMETRY.hand_crop_px)
            if hcp is not None:
                nm = f"{stem}_h{j}.jpg"; cv2.imwrite(str(out / "hand" / nm), hcp)
                hands.append(nm)
        if hn is None and len(hands) < 2:
            skip["no usable roi"] += 1; continue
        rows.append({"stem": stem, "site": site, "head": hn, "hands": hands,
                     "split": "val" if site in val_sites else "train", "labels": labels})
    if n % 300 == 0:
        print(f"  {n}/{len(items)}  kept {len(rows)}", flush=True)

(out / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
print(f"\n{len(rows)} person records -> {out}")
print("skipped:", dict(skip))
for split in ("train", "val"):
    sub = [x for x in rows if x["split"] == split]
    print(f"{split}: {len(sub)}  head-crops {sum(1 for x in sub if x['head'])}  "
          f"two-hands {sum(1 for x in sub if len(x['hands'])==2)}")
    for at in ATTRS:
        c = collections.Counter(x["labels"][at] for x in sub)
        print(f"   {at:9} present {c['present']:5d}  absent {c['absent']:5d}  unclear {c['unclear']:5d}")
