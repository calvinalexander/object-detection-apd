"""Does tracking cost frame rate, and if so which part of it?

BoT-SORT's default config sets `gmc_method: sparseOptFlow` -- global motion
compensation, which runs optical flow on every frame so that tracking survives a
MOVING camera. These are fixed CCTV cameras bolted to a kitchen ceiling, so that
computation has nothing to correct and is pure cost.

Timed separately from the classifiers, because the pose call is the only place
tracking can charge anything: association happens after detection and the
classifier stage is identical either way.
"""
import argparse, statistics, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/recording/batch_3/Recording 2026-09-11 135526.mp4")
ap.add_argument("--frames", type=int, default=120)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.40)
ap.add_argument("--weights", default=".work/yolo11x-pose.pt")
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
frames = []
while len(frames) < a.frames:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
print(f"{Path(a.video).name}: {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, "
      f"{Path(a.weights).name} @ {a.imgsz}\n")

MODES = [
    ("predict (no tracking)",   None,                               None),
    ("botsort (default, GMC)",  "botsort.yaml",                     None),
    ("botsort (gmc=none)",      ".work/trackers/botsort_nogmc.yaml", None),
    ("bytetrack",               "bytetrack.yaml",                   None),
]

print(f"{'mode':26} {'ms/frame':>9} {'fps':>7} {'vs predict':>11}  {'ids':>5} {'persons/frame':>14}")
base = None
for name, tracker, _ in MODES:
    m = YOLO(a.weights)
    for f in frames[:5]:                                   # warm up
        (m.track(f, imgsz=a.imgsz, conf=a.conf, persist=True, tracker=tracker,
                 verbose=False, device=0) if tracker else
         m.predict(f, imgsz=a.imgsz, conf=a.conf, verbose=False, device=0))
    m = YOLO(a.weights)                                    # fresh tracker state
    ids, counts, times = set(), [], []
    for f in frames:
        t = time.perf_counter()
        r = (m.track(f, imgsz=a.imgsz, conf=a.conf, persist=True, tracker=tracker,
                     verbose=False, device=0)[0] if tracker else
             m.predict(f, imgsz=a.imgsz, conf=a.conf, verbose=False, device=0)[0])
        times.append((time.perf_counter() - t) * 1000)
        counts.append(0 if r.boxes is None else len(r.boxes))
        if tracker and r.boxes is not None and r.boxes.id is not None:
            ids |= set(r.boxes.id.int().cpu().tolist())
    ms = statistics.median(times)
    if base is None: base = ms
    print(f"{name:26} {ms:9.1f} {1000/ms:7.1f} {ms/base-1:+10.1%}  "
          f"{len(ids) if tracker else 0:5d} {np.mean(counts):14.2f}")
