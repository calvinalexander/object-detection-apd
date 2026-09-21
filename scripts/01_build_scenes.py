"""Assign every frame a camera id and a finer scene id.

Two levels, because they answer different questions. `camera_id` groups frames by
the room and viewing angle and is what splits must be grouped on -- production runs
at other sites, so the metric that matters is performance on a camera never trained
on. `scene_id` is the finer pixel-level cluster, still useful for de-duplicating
near-identical frames within one camera.
"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.scenes import hash_directory, hash_cameras, cluster, DEFAULT_THRESHOLD

CAMERA_THRESHOLD = 6      # smallest value that merges frames verified to be one camera
ROOT = Path("data/sppg"); OUT = Path("artifacts/scenes.json")

names, packed = hash_directory(ROOT / ".thumbnail")
scene_ids = cluster(packed, DEFAULT_THRESHOLD)
cam_names, cam_packed = hash_cameras(ROOT / ".thumbnail")
camera_ids = cluster(cam_packed, CAMERA_THRESHOLD)
cam_of = {n: int(i) for n, i in zip(cam_names, camera_ids)}

OUT.write_text(json.dumps({
    "scene_threshold": DEFAULT_THRESHOLD, "camera_threshold": CAMERA_THRESHOLD,
    "n_frames": len(names),
    "n_scenes": len(set(scene_ids.tolist())), "n_cameras": len(set(cam_of.values())),
    "frame_to_scene": {n: int(s) for n, s in zip(names, scene_ids)},
    "frame_to_camera": cam_of,
}, indent=1))
cs = collections.Counter(cam_of.values())
print(f"{len(names)} frames -> {len(set(scene_ids.tolist()))} scenes, {len(cs)} cameras")
print(f"largest cameras: {[c for _, c in cs.most_common(6)]} "
      f"= {100*sum(c for _,c in cs.most_common(6))/len(names):.0f}% of pool")
print(f"wrote {OUT}")
