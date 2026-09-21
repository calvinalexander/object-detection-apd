"""Build the person-crop dataset that the deployable model trains on.

The labelling stack -- pose@1280 + DINOv2-base + three classifiers -- runs at
about 8 img/s on a full GPU. That is fine for labelling, which is not realtime,
and hopeless for the edge. This distils it into: one person detector, then one
small multi-head classifier on the person crop.

Per-person verdicts are what the consensus actually validated (96.6-99.3% where
both our voters agree), so they are the supervision. PPE box geometry is NOT
used: half of it is apd_mbg's and half is a generous anatomical ROI whose IoU
with the real equipment is .09-.21, which would teach localisation we cannot
verify. `unclear` is preserved as a third state and masked out of the loss
rather than folded into `absent`.

Split is by SITE, so the holdout measures a new kitchen and not a new frame of
a kitchen already seen.
"""
import argparse, collections, json, random, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.overlay import clean
from apd.schema import Attribute, assert_contract

assert_contract()
ATTRS = [a.value for a in Attribute]
ap = argparse.ArgumentParser()
ap.add_argument("--consensus", default="artifacts/consensus_ci3b.jsonl")
ap.add_argument("--out", default="data/distill")
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--min-side", type=int, default=40)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()

recs = [json.loads(l) for l in Path(a.consensus).read_text().splitlines()]
sites = sorted({r["site"] for r in recs})
rng = random.Random(a.seed); rng.shuffle(sites)
val_sites = set(sites[:max(1, int(len(sites) * a.val_frac))])
print(f"{len(recs)} frames, {len(sites)} sites, {len(val_sites)} held out for val")

out = Path(a.out); (out / "crops").mkdir(parents=True, exist_ok=True)
rows, skipped = [], collections.Counter()
for r in recs:
    src = Path("data/sites") / r["site"] / f"{r['id']}.jpg"
    if not src.exists():
        skipped["frame missing"] += 1; continue
    img = None
    for p in r["persons"]:
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        if min(x2 - x1, y2 - y1) < a.min_side:
            skipped["person too small"] += 1; continue
        labels = {k: p["attrs"][k]["decision"] for k in ATTRS}
        if all(v == "unclear" for v in labels.values()):
            skipped["nothing decided"] += 1; continue
        if img is None:
            img = clean(cv2.imread(str(src)))
            if img is None:
                skipped["unreadable"] += 1; break
        pad = int(0.10 * max(x2 - x1, y2 - y1))
        crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if crop.size == 0:
            skipped["empty crop"] += 1; continue
        name = f"{r['site']}__{r['id']}_{p['person']}.jpg"
        cv2.imwrite(str(out / "crops" / name), crop)
        rows.append({"crop": name, "site": r["site"], "image": str(src),
                     "split": "val" if r["site"] in val_sites else "train",
                     "labels": labels})

(out / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
print(f"\n{len(rows)} person crops written to {out/'crops'}")
print("skipped:", dict(skipped))
for split in ("train", "val"):
    sub = [x for x in rows if x["split"] == split]
    print(f"\n{split}: {len(sub)} crops")
    for at in ATTRS:
        c = collections.Counter(x["labels"][at] for x in sub)
        print(f"   {at:9} present {c['present']:5d}  absent {c['absent']:5d}  unclear {c['unclear']:5d}")
