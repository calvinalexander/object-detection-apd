"""Run pose over every frame and materialise the person/ROI manifest."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ultralytics import YOLO
from apd.schema import GEOMETRY, assert_contract
from apd.pose import build_rois

assert_contract()
ROOT = Path("data/sppg")
OUT = Path("artifacts/persons.jsonl")
WEIGHTS = ".work/yolo11x-pose.pt"

scenes = json.loads(Path("artifacts/scenes.json").read_text())["frame_to_scene"]
frames = sorted(p for p in ROOT.iterdir() if p.is_file())
model = YOLO(WEIGHTS)

stats = collections.Counter()
with OUT.open("w") as fh:
    for n, path in enumerate(frames, 1):
        r = model.predict(str(path), imgsz=GEOMETRY.pose_imgsz, conf=GEOMETRY.pose_conf,
                          verbose=False, device=0)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            stats["frames_without_person"] += 1
            continue
        kps = r.keypoints.data.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        rois = build_rois(r.boxes.xyxy.cpu().numpy(), kps)
        for idx, p in enumerate(rois):
            if p.head is None and not p.hands:
                stats["persons_without_roi"] += 1
                continue
            stats["persons"] += 1
            stats["head_localizable"] += p.head_localizable
            stats["frontal_face"] += p.frontal_face
            stats["head_collision"] += p.head_collision
            fh.write(json.dumps({
                "person_id": f"{path.stem}#{idx}",
                "frame": path.name,
                "scene_id": scenes.get(path.name, -1),
                "person_box": [round(float(v), 1) for v in p.person_box],
                "head_roi": [round(float(v), 1) for v in p.head] if p.head else None,
                "hand_rois": [[round(float(v), 1) for v in h] for h in p.hands],
                "head_localizable": bool(p.head_localizable),
                "frontal_face": bool(p.frontal_face),
                "head_collision": bool(p.head_collision),
                "person_conf": round(float(confs[idx]), 3),
                # Max confidence over nose/eyes -- the evidence that a face is
                # actually pointed at the camera, as opposed to keypoints the
                # model hallucinated onto the back of a head or onto no person.
                "face_kp_conf": round(float(max(kps[idx][i][2] for i in (0, 1, 2))), 3),
                # Nose, eyes, ears, shoulders as (x, y, conf). Stored so the
                # frontality test can be re-tuned without re-running inference --
                # keypoint confidence turned out to mean "I know where this joint
                # is", not "this joint faces the camera", so the gate has to be
                # geometric and geometry needs the coordinates.
                # 0-10: nose, eyes, ears, shoulders, elbows, wrists. Wrists were
                # omitted before, which meant hand-ROI quality could not be
                # audited or gated without re-running inference.
                "kp": [[round(float(v), 1) for v in kps[idx][i][:2]]
                       + [round(float(kps[idx][i][2]), 3)] for i in range(11)],
            }) + "\n")
        if n % 250 == 0:
            print(f"  {n}/{len(frames)} frames, {stats['persons']} persons", flush=True)

n = stats["persons"]
print(f"\n{len(frames)} frames -> {n} person instances")
print(f"  head locatable : {stats['head_localizable']}/{n} = {100*stats['head_localizable']/n:.1f}%"
      f"  (gates head_cap)")
print(f"  facing camera  : {stats['frontal_face']}/{n} = {100*stats['frontal_face']/n:.1f}%"
      f"  (gates mask -- rest auto-'unclear')")
print(f"  head collision : {stats['head_collision']}/{n} = {100*stats['head_collision']/n:.1f}%"
      f"  (routed to human review, not auto-labelled)")
print(f"  frames with no person: {stats['frames_without_person']}")
print(f"wrote {OUT}")
