"""Freeze one test set, shared by every model, grouped so it cannot leak.

Three leakage modes have now been found in this project, and a test set has to
survive all of them:

  cross-model   model A's validation split was model B's training data. Fixed by
                having ONE test set that every model excludes, not one per run.
  same-site     a model can memorise a kitchen. Fixed by holding out whole site
                codes.
  adjacent-frame  the crops named `frame_000086` are consecutive frames pulled
                from video. 57% of head_cap val crops sit within ONE frame of a
                training crop and 96% within five -- different filenames, the
                same picture. Fixed by holding out contiguous BLOCKS of frames
                with a margin, never individual frames.

Writes a frozen list of stems with a hash. Training code excludes it; nothing
tunes against it. If the hash changes, every number measured against it is void.
"""
import argparse, glob, hashlib, json, re, sys
import numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--roots", default="data/crops_ci_inpaint,data/crops_ci3,data/crops_cls")
ap.add_argument("--frac", type=float, default=0.28)
ap.add_argument("--block", type=int, default=150, help="video frames per group")
ap.add_argument("--margin", type=int, default=30, help="frames dropped either side of a held-out block")
ap.add_argument("--seed", type=int, default=20260914)
ap.add_argument("--out", default="data/splits/frozen_test.json")
a = ap.parse_args()

VID = re.compile(r"^[0-9a-f]{8}-(.+?)_(\d+)(?:_h\d)?$")

# The dated SPPG frames carry no group in their filename, so each would be its
# own group and a test frame could sit beside a training frame from the SAME
# fixed camera. apd.scenes.camera_hash exists for this: it signs the room rather
# than the moment (perceptual hashing measured pixel difference instead, which
# once let 82% of "unseen scene" frames come from cameras present in training).
_CAM: dict[str, str] = {}
def _load_cameras(threshold: int = 6):
    from apd.scenes import hash_cameras
    d = Path("data/sppg")
    if not d.is_dir():
        return
    names, H = hash_cameras(d)
    parent = list(range(len(names)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if int(np.count_nonzero(H[i] != H[j])) <= threshold:
                a_, b_ = find(i), find(j)
                if a_ != b_:
                    parent[a_] = b_
    for i, n in enumerate(names):
        _CAM[Path(n).stem] = f"cam{find(i):03d}"
    print(f"camera grouping: {len(names)} sppg frames -> {len(set(_CAM.values()))} cameras")

def stem_of(p):
    s = Path(p).stem
    for suf in ("__o", "__h", "__g"):
        s = s.split(suf)[0]
    return s

def group_of(stem):
    """The unit that must not be split across train and test."""
    if "__" in stem:                                  # site crop
        return ("site", stem.split("__")[0])
    m = VID.match(stem)
    if m:
        src, idx = m.group(1), m.group(2)
        if src.startswith("frame_") or "frame_" in src:
            n = int(re.sub(r"\D", "", src) or 0)
            return ("video", f"blk{n // a.block:04d}")
        return ("clip", src)
    # dated SPPG frame: group by camera, falling back to the frame itself
    base = stem.rsplit("_", 1)[0] if re.search(r"_\d+(?:_h\d)?$", stem) else stem
    base = re.sub(r"_h\d$", "", base).rsplit("_", 1)[0] if base.endswith(("_h0", "_h1")) else base
    for key in (base, stem):
        if key in _CAM:
            return ("camera", _CAM[key])
    return ("other", stem)

_load_cameras()

# ---- inventory every crop any dataset knows about
labels, groups = {}, defaultdict(set)
for root in a.roots.split(","):
    for attr in [x.value for x in Attribute]:
        for split in ("train", "val"):
            for cls in ("present", "absent"):
                for p in glob.glob(f"{root}/{attr}/{split}/{cls}/*.jpg"):
                    st = stem_of(p)
                    labels.setdefault((attr, st), cls)
                    groups[group_of(st)].add((attr, st))

import random
rng = random.Random(a.seed)
keys = sorted(groups, key=lambda g: (g[0], g[1]))
rng.shuffle(keys)

total = len(labels)
test_groups, taken = [], 0
for g in keys:
    if taken >= a.frac * total:
        break
    test_groups.append(g)
    taken += len(groups[g])

test_set = {g: sorted(groups[g]) for g in test_groups}
test_stems = {st for v in test_set.values() for _, st in v}

# ---- margin: a training frame adjacent to a held-out block is still the same
# picture, so blocks either side of a held-out video block are dropped from BOTH.
held_blocks = {int(g[1][3:]) for g in test_groups if g[0] == "video"}
quarantine = set()
for g, members in groups.items():
    if g[0] != "video" or g in test_groups:
        continue
    b = int(g[1][3:])
    if any(abs(b - h) * a.block <= a.margin + a.block for h in held_blocks):
        quarantine |= {st for _, st in members}

payload = {
    "created": "2026-09-14",
    "params": {"frac": a.frac, "block": a.block, "margin": a.margin, "seed": a.seed},
    "groups": {f"{k[0]}:{k[1]}": [f"{at}/{st}" for at, st in v] for k, v in test_set.items()},
    "test_stems": sorted(test_stems),
    "quarantine_stems": sorted(quarantine),
}
body = json.dumps(payload, sort_keys=True).encode()
payload["sha256"] = hashlib.sha256(body).hexdigest()
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=1))

print(f"frozen test set -> {out}   sha256 {payload['sha256'][:16]}")
print(f"{len(test_stems)} stems held out of {len({st for _, st in labels})} "
      f"({len(test_stems)/max(1,len({st for _,st in labels})):.0%}), "
      f"{len(test_groups)} groups, plus {len(quarantine)} quarantined by margin\n")
print(f"{'attr':9} {'test crops':>11} {'present':>8} {'absent':>7} {'groups':>7}")
for attr in [x.value for x in Attribute]:
    mem = [(at, st) for v in test_set.values() for at, st in v if at == attr]
    pres = sum(1 for k in mem if labels[k] == "present")
    gs = len({group_of(st) for _, st in mem})
    print(f"{attr:9} {len(mem):11d} {pres:8d} {len(mem)-pres:7d} {gs:7d}")
kinds = defaultdict(int)
for g in test_groups:
    kinds[g[0]] += len(groups[g])
print("\ncomposition:", dict(kinds))
