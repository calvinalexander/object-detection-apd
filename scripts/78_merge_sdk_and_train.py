"""Add the SDK crops to training and retrain, keeping the stopping signal honest.

SDK crops are not all dumped into train: 15% of their GROUPS go to the internal
validation split, so early stopping is judged partly on the new domain rather
than entirely on the old one. Grouping is by source clip / frame block, because
these are video frames -- a random split would put adjacent frames of the same
person on both sides and stop on a near-duplicate.
"""
import argparse, glob, random, re, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="data/crops_cocoheld")
ap.add_argument("--sdk", default="data/crops_sdk")
ap.add_argument("--out", default="data/crops_plus_sdk")
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
rng = random.Random(a.seed)

def group_of(name):
    """Source clip, or a block of 150 video frames -- never a single frame."""
    s = name.replace("sdk_", "")
    m = re.match(r"^[0-9a-f]{8}-(cam\d+_[\d\-_]+)_frame", s)
    if m: return "clip:" + m.group(1)
    m = re.match(r"^[0-9a-f]{8}-frame_(\d+)", s)
    if m: return "blk:%04d" % (int(m.group(1)) // 150)
    return "other:" + s[:12]

out = Path(a.out)
if out.exists(): shutil.rmtree(out)
for attr in ("head_cap", "mask", "gloves"):
    # carry the existing split across unchanged
    kept = {"train": 0, "val": 0}
    for sp in ("train", "val"):
        for cls in ("present", "absent"):
            d = out / attr / sp / cls; d.mkdir(parents=True, exist_ok=True)
            for p in glob.glob(f"{a.base}/{attr}/{sp}/{cls}/*.jpg"):
                shutil.copy(p, d / Path(p).name); kept[sp] += 1

    items = [(p, cls) for cls in ("present", "absent")
             for p in glob.glob(f"{a.sdk}/{attr}/{cls}/*.jpg")]
    groups = sorted({group_of(Path(p).stem) for p, _ in items})
    rng.shuffle(groups)
    vg = set(groups[:int(len(groups) * a.val_frac)]) if a.val_frac > 0 else set()
    added = {"train": 0, "val": 0}
    for p, cls in items:
        sp = "val" if group_of(Path(p).stem) in vg else "train"
        shutil.copy(p, out / attr / sp / cls / Path(p).name); added[sp] += 1
    n = {sp: len(glob.glob(f"{out}/{attr}/{sp}/*/*.jpg")) for sp in ("train", "val")}
    print(f"{attr:9} base {kept['train']}+{kept['val']} | SDK +{added['train']} train "
          f"+{added['val']} val ({len(groups)} groups) | total train {n['train']}, val {n['val']}")
print(f"\nwrote {out}")
