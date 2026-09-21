"""Freeze the evaluation splits. Run once, before any model exists.

Three scenes hold 55% of the pool, which forces a choice. A strictly
scene-disjoint test set cannot contain the dominant camera views without
swallowing a quarter of the data; a test set that does contain them cannot be
scene-disjoint. Rather than fudge one number, this writes two test sets that
measure two different things:

  test_unseen_camera   whole scenes never seen in training. Answers "does it work
                      on a camera view we did not train on". This is the headline
                      metric and the honest one.

  test_same_camera    the temporally-latest frames of the dominant cameras, held
                      out of training. Answers "does it keep working on a camera
                      we did train on, later in the day". Weaker evidence -- the
                      frames are near-duplicates of training frames -- so it is
                      reported separately and never averaged into the headline.

Frame order comes from the violation-record id embedded in each filename, which
increases monotonically through the capture day.
"""
import json, re, sys, hashlib, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCENES = json.loads(Path("artifacts/scenes.json").read_text())
# Group on CAMERA, not scene. Grouping on the pixel-level scene id left 82% of
# "unseen scene" test frames on cameras that were also in training, so the split
# was measuring same-camera performance -- the opposite of the deployment case.
FRAME_TO_SCENE = SCENES["frame_to_camera"]
OUT = Path("artifacts/splits.json")

LARGE_SCENE_MIN = 120     # cameras at or above this are "dominant"
UNSEEN_TARGET = 0.16      # share of all frames for test_unseen_camera
VAL_TARGET = 0.14
TAIL_SHARE = 0.15         # latest fraction of each dominant scene held out

def record_id(name: str) -> int:
    m = re.match(r"\d{4}-\d{2}-\d{2}_(\d+)-", name)
    return int(m.group(1)) if m else 0

by_scene = collections.defaultdict(list)
for frame, scene in FRAME_TO_SCENE.items():
    by_scene[scene].append(frame)
for frames in by_scene.values():
    frames.sort(key=record_id)

total = len(FRAME_TO_SCENE)
large = {s for s, f in by_scene.items() if len(f) >= LARGE_SCENE_MIN}
small = sorted((s for s in by_scene if s not in large), key=lambda s: -len(by_scene[s]))

# Whole small scenes -> unseen test, then val. Largest-first so the targets are
# met with as few scenes as possible, leaving the long tail for training.
assign, want_unseen, want_val = {}, UNSEEN_TARGET * total, VAL_TARGET * total
got_unseen = got_val = 0
for i, scene in enumerate(small):
    n = len(by_scene[scene])
    if got_unseen < want_unseen and i % 2 == 0:
        assign[scene] = "test_unseen_camera"; got_unseen += n
    elif got_val < want_val:
        assign[scene] = "val"; got_val += n
    else:
        assign[scene] = "train"

splits = collections.defaultdict(list)
for scene, frames in by_scene.items():
    if scene in large:
        cut = int(len(frames) * (1 - TAIL_SHARE))
        splits["train"].extend(frames[:cut])
        splits["test_same_camera"].extend(frames[cut:])
    else:
        splits[assign[scene]].extend(frames[:])

for k in splits:
    splits[k].sort()

train_scenes = {FRAME_TO_SCENE[f] for f in splits["train"]}
unseen_cameras = {FRAME_TO_SCENE[f] for f in splits["test_unseen_camera"]}
val_scenes = {FRAME_TO_SCENE[f] for f in splits["val"]}
assert not (train_scenes & unseen_cameras), "camera leak: train <-> test_unseen_camera"
assert not (train_scenes & val_scenes), "camera leak: train <-> val"
assert sum(len(v) for v in splits.values()) == total, "frames lost"
assert not set(splits["train"]) & set(splits["test_same_camera"]), "frame leak"

payload = {k: sorted(v) for k, v in splits.items()}
digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
OUT.write_text(json.dumps({
    "frozen_sha256": digest,
    "params": {"large_scene_min": LARGE_SCENE_MIN, "tail_share": TAIL_SHARE},
    "splits": payload,
}, indent=1))

print(f"{'split':<22}{'frames':>8}{'cameras':>8}{'share':>8}")
for k in ("train", "val", "test_unseen_camera", "test_same_camera"):
    sc = len({FRAME_TO_SCENE[f] for f in splits[k]})
    print(f"{k:<22}{len(splits[k]):>8}{sc:>8}{100*len(splits[k])/total:>7.1f}%")
print(f"\ndominant cameras (>={LARGE_SCENE_MIN} frames): {len(large)}")
print("scene-disjointness train<->val and train<->test_unseen_camera: verified")
print(f"frozen sha256: {digest}")
print(f"wrote {OUT}")
