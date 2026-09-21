"""One training set for the architecture bake-off, with the eval frames removed.

Train = every crop NOT derived from the 53 human-annotated frames (site crops,
apd_v3 val crops, the rest of SPPG). An internal validation split is carved from
it by SITE for early stopping -- carving it at random would put near-duplicate
crops of the same person on both sides, and stopping on that is how the earlier
runs reached a 0.0003 training loss while generalising badly.
"""
import argparse, glob, json, random, re, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/crops_ci_inpaint")
ap.add_argument("--out", default="data/crops_cocoheld")
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()

coco = json.loads(Path("data/sppg/.exports/coco-1778210446.4251382.json").read_text())
imgs = {i["id"]: i["file_name"] for i in coco["images"]}
ANN = {Path(imgs[x["image_id"]]).stem for x in coco["annotations"]}

def stem_of(p):
    s = Path(p).stem
    for suf in ("__o", "__h", "__g"):
        s = s.split(suf)[0]
    return s
def from_ann(s):
    return s in ANN or re.sub(r"_\d+(_h\d)?$", "", s) in ANN
def group_of(s):
    if "__" in s: return "site:" + s.split("__")[0]
    m = re.match(r"^[0-9a-f]{8}-(.*?)_\d+(?:_h\d)?$", s)
    if m and "frame_" in m.group(1):
        return "vid:blk%04d" % (int(re.sub(r"\D", "", m.group(1)) or 0) // 150)
    return "sppg:" + re.sub(r"_\d+(_h\d)?$", "", s)[:24]

out = Path(a.out)
if out.exists(): shutil.rmtree(out)
rng = random.Random(a.seed)
for attr in ("head_cap", "mask", "gloves"):
    items, dropped = [], 0
    for sp in ("train", "val"):
        for cls in ("present", "absent"):
            for p in glob.glob(f"{a.src}/{attr}/{sp}/{cls}/*.jpg"):
                s = stem_of(p)
                if from_ann(s): dropped += 1; continue
                items.append((p, cls, group_of(s)))
    groups = sorted({g for _, _, g in items}); rng.shuffle(groups)
    nval = max(1, int(len(groups) * a.val_frac))
    vg = set(groups[:nval])
    counts = {}
    for p, cls, g in items:
        sp = "val" if g in vg else "train"
        d = out / attr / sp / cls
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, d / Path(p).name)
        counts[(sp, cls)] = counts.get((sp, cls), 0) + 1
    print(f"{attr:9} dropped {dropped:5d} (annotated frames) | "
          f"train {counts.get(('train','present'),0)}+/{counts.get(('train','absent'),0)}- | "
          f"val {counts.get(('val','present'),0)}+/{counts.get(('val','absent'),0)}- | "
          f"{len(groups)} groups, {nval} held for val")
print(f"\nwrote {out}")
