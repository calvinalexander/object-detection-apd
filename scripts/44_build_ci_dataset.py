"""Colour-invariant crop dataset, now including the mined site crops.

Two things are being fixed at once and they are separate:

  colour  -- global hue rotation (the whole crop rotated together, so the
             skin-vs-object contrast that distinguishes a glove or a mask is
             preserved while the absolute hue is destroyed). Grayscale copies
             for head_cap only, where colour carried no legitimate signal.
  coverage -- crops_cls is 89-95% one camera in one room. The site crops add
             60 sites; without them the classifier extrapolates rather than
             recognises, which is what put a blue cap at p=0.023.

Val is split by SITE, never by crop: a site in train must not appear in val,
or the score measures memorisation of a room.
"""
import argparse, glob, json, random, shutil, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--attribute", default=Attribute.HEAD_CAP.value)
ap.add_argument("--out", default="data/crops_ci2")
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--site-root", default="artifacts/site_crops_clean")
args = ap.parse_args()
rng = random.Random(args.seed)
attr = args.attribute

def hue_shift(img, deg):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h[..., 0] = (h[..., 0].astype(int) + deg // 2) % 180
    return cv2.cvtColor(h, cv2.COLOR_HSV2BGR)

def gray(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

# --- gather originals with a group key -----------------------------------
items = []   # (path, label, group)
for p in glob.glob(f"data/crops_cls/{attr}/*/*/*.jpg"):
    label = Path(p).parent.name
    stem = Path(p).stem.split("-", 1)[-1]
    items.append((p, label, "sppg:" + stem.rsplit("_", 1)[0]))

# COCO seed annotations from data/sppg/.exports/coco-1778210446.4251382.json
seed_store = Path("artifacts/seed_labels.jsonl")
n_coco = 0
if seed_store.exists():
    for line in seed_store.read_text().splitlines():
        r = json.loads(line)
        pid = r["person_id"].replace("#", "_")
        frame = r["frame"]
        scene_grp = f"coco:{r.get('scene_id', frame)}"
        if attr in (Attribute.HEAD_CAP.value, Attribute.MASK.value):
            v = r["labels"].get(attr)
            if v in ("present", "absent"):
                cp = Path(f"artifacts/crops/head/{pid}.jpg")
                if cp.exists():
                    items.append((str(cp), v, scene_grp))
                    n_coco += 1
        elif attr == Attribute.GLOVES.value:
            for h in r.get("hands", []):
                v = h.get("value")
                if v in ("present", "absent"):
                    cp = Path(f"artifacts/crops/hand/{pid}_h{h['hand']}.jpg")
                    if cp.exists():
                        items.append((str(cp), v, scene_grp))
                        n_coco += 1

site_store = Path(f"{args.site_root}/{attr}/labels.jsonl")
n_site = 0
if site_store.exists():
    for line in site_store.read_text().splitlines():
        r = json.loads(line)
        if r["value"] == "unclear":       # abstention is not a class
            continue
        items.append((f"{args.site_root}/{attr}/pool/{r['crop']}",
                      r["value"], "site:" + r["site"]))
        n_site += 1

print(f"{attr}: {len(items)} decided crops ({len(items)-n_site-n_coco} cls + {n_coco} COCO seed + {n_site} site), "
      f"{len({g for _,_,g in items})} groups")

groups = sorted({g for _, _, g in items})
rng.shuffle(groups)
n_val = max(1, int(len(groups) * args.val_frac))
val_groups = set(groups[:n_val])

out = Path(args.out) / attr
if out.exists():
    shutil.rmtree(out)
for s in ("train", "val"):
    for c in ("present", "absent"):
        (out / s / c).mkdir(parents=True, exist_ok=True)

counts = {}
for path, label, group in items:
    img = cv2.imread(path)
    if img is None:
        continue
    split = "val" if group in val_groups else "train"
    stem = Path(path).stem
    variants = [("o", img)]
    if split == "train":                       # never augment val
        variants += [(f"h{d}", hue_shift(img, d)) for d in (45, 90, 135)]
        if attr == Attribute.HEAD_CAP.value:
            variants.append(("g", gray(img)))
    for tag, v in variants:
        cv2.imwrite(str(out / split / label / f"{stem}__{tag}.jpg"), v)
    counts[(split, label)] = counts.get((split, label), 0) + len(variants)

print(f"val groups: {n_val}/{len(groups)}")
for s in ("train", "val"):
    for c in ("present", "absent"):
        n = counts.get((s, c), 0)
        real = len(list((out / s / c).glob("*__o.jpg")))
        print(f"  {s}/{c}: {n:5d} images ({real} original)")
print(f"wrote {out}")
