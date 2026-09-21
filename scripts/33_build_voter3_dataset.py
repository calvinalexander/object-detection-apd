"""Assemble a training set for a third voter from the two shipped val splits.

apd_v2/val and apd_v3/val are 194 distinct images once deduplicated by content
hash (14 appear in both). That is the only 7-class annotated data we have,
negatives included.

Training on it costs us the clean evaluation set, so the split here is
camera-grouped rather than random -- the same lesson that invalidated the earlier
SPPG splits, where a perceptual-hash "scene" id let one camera sit on both sides
and inflated every metric. Grouping on a blur-hash camera signature keeps a
holdout that actually measures generalisation.
"""
import collections, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.scenes import camera_hash, cluster

NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
SRC = Path("data/dump/dataset")
OUT = Path("data/voter3")
HOLDOUT_FRAC = 0.22

pairs = {}
for m in ("apd_v2", "apd_v3"):
    for img in sorted((SRC/m/"dataset/images/val").iterdir()):
        if img.suffix.lower() not in (".jpg",".jpeg",".png"): continue
        lab = SRC/m/"dataset/labels/val"/(img.stem+".txt")
        if not lab.exists(): continue
        h = hashlib.md5(img.read_bytes()).hexdigest()
        pairs.setdefault(h, (img, lab))          # first wins; duplicates dropped
items = list(pairs.values())
print(f"{len(items)} distinct image/label pairs from apd_v2 + apd_v3 val")

H = np.packbits(np.array([camera_hash(i) for i,_ in items]), axis=1)
cam = cluster(H, 6)
sizes = collections.Counter(cam.tolist())
print(f"camera clusters: {len(sizes)}  largest {max(sizes.values())}")

# Hold out whole cameras, largest-first so the target is met with few groups.
order = [c for c,_ in sizes.most_common()]
want = int(len(items) * HOLDOUT_FRAC)
hold, got = set(), 0
for c in order[::-1]:                            # smallest cameras first: more variety held out
    if got >= want: break
    hold.add(c); got += sizes[c]
split = ["val" if c in hold else "train" for c in cam]
print(f"train {split.count('train')}  holdout {split.count('val')} "
      f"({len(hold)} cameras held out of {len(sizes)})")

if OUT.exists(): shutil.rmtree(OUT)
for s in ("train","val"):
    (OUT/"images"/s).mkdir(parents=True); (OUT/"labels"/s).mkdir(parents=True)
cnt = collections.Counter()
for (img,lab), s in zip(items, split):
    shutil.copy2(img, OUT/"images"/s/img.name)
    shutil.copy2(lab, OUT/"labels"/s/(img.stem+".txt"))
    for line in lab.read_text().splitlines():
        f = line.split()
        if f: cnt[(s,int(f[0]))] += 1

(OUT/"dataset.yaml").write_text(
    "path: " + str(OUT.resolve()) + "\ntrain: images/train\nval: images/val\n"
    f"nc: {len(NAMES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i,n in enumerate(NAMES)))
json.dump({"camera_of": {items[i][0].name: int(cam[i]) for i in range(len(items))},
           "holdout_cameras": sorted(hold)}, open(OUT/"cameras.json","w"), indent=1)

print(f"\n{'class':<14}{'train':>7}{'holdout':>9}")
for i,n in enumerate(NAMES):
    print(f"{n:<14}{cnt[('train',i)]:>7}{cnt[('val',i)]:>9}")
print(f"\nwrote {OUT}/dataset.yaml")
