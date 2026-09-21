"""Benchmark the tracker that will actually ship, not only Ultralytics' built-in.

Every mode measured so far went through `pose.track()`, which exists only inside
the YOLO wrapper. Production runs the vendored upstream ByteTrack in
/data/adin/sppg-ai/src/apd_tracker as a separate stage fed with boxes, because
the Axelera runtime emits detections and nothing else. This times that path:
`predict()` for boxes, then `BYTETracker.update()`, which is what an edge
deployment would do regardless of inference backend.
"""
import argparse, statistics, sys, time, types
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/data/adin/sppg-ai/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd_tracker.byte_tracker import BYTETracker

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/recording/batch_3/Recording 2026-09-11 135526.mp4")
ap.add_argument("--frames", type=int, default=120)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.40)
ap.add_argument("--weights", default=".work/yolo11x-pose.pt")
a = ap.parse_args()

# Production's tracker args (apd_monitor_axelera_multi / standalone_detector)
prod_args = types.SimpleNamespace(track_thresh=0.30, track_buffer=60, match_thresh=0.8,
                                  aspect_ratio_thresh=1.6, min_box_area=10, mot20=False)

cap = cv2.VideoCapture(a.video)
frames = []
while len(frames) < a.frames:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
H, W = frames[0].shape[:2]
print(f"{Path(a.video).name}: {len(frames)} frames {W}x{H}, {Path(a.weights).name} @ {a.imgsz}\n")

def bench(label, tracker_yaml=None, external=False):
    m = YOLO(a.weights)
    for f in frames[:5]:
        (m.track(f, imgsz=a.imgsz, conf=a.conf, persist=True, tracker=tracker_yaml,
                 verbose=False, device=0) if tracker_yaml else
         m.predict(f, imgsz=a.imgsz, conf=a.conf, verbose=False, device=0))
    m = YOLO(a.weights)
    ext = BYTETracker(prod_args) if external else None
    det_ms, trk_ms, ids, counts = [], [], set(), []
    for f in frames:
        t0 = time.perf_counter()
        if tracker_yaml:
            r = m.track(f, imgsz=a.imgsz, conf=a.conf, persist=True, tracker=tracker_yaml,
                        verbose=False, device=0)[0]
        else:
            r = m.predict(f, imgsz=a.imgsz, conf=a.conf, verbose=False, device=0)[0]
        t1 = time.perf_counter()
        n = 0 if r.boxes is None else len(r.boxes)
        counts.append(n)
        if tracker_yaml and r.boxes is not None and r.boxes.id is not None:
            ids |= set(r.boxes.id.int().cpu().tolist())
        if external:
            if n:
                xy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xy, cf]).astype(np.float32)
            else:
                dets = np.zeros((0, 5), np.float32)
            t2 = time.perf_counter()
            for tr in ext.update(dets, (H, W), (H, W)):
                tid = getattr(tr, "track_id", None)
                if tid is not None: ids.add(int(tid))
            trk_ms.append((time.perf_counter() - t2) * 1000)
        det_ms.append((t1 - t0) * 1000)
    d = statistics.median(det_ms)
    t = statistics.median(trk_ms) if trk_ms else 0.0
    return label, d, t, len(ids), float(np.mean(counts))

rows = [
    bench("predict only (no tracking)"),
    bench("predict + apd_tracker (PRODUCTION)", external=True),
    bench("ultralytics bytetrack", "bytetrack.yaml"),
    bench("ultralytics botsort gmc=none", ".work/trackers/botsort_nogmc.yaml"),
    bench("ultralytics botsort (default GMC)", "botsort.yaml"),
]
base = rows[0][1] + rows[0][2]
print(f"{'mode':38} {'detect':>8} {'track':>7} {'total':>8} {'fps':>7} {'overhead':>9} {'ids':>5}")
for label, d, t, nid, np_ in rows:
    tot = d + t
    print(f"{label:38} {d:8.1f} {t:7.2f} {tot:8.1f} {1000/tot:7.1f} {tot/base-1:+8.1%} {nid:5d}")
print(f"\npersons/frame {rows[0][4]:.2f}   (detect and track columns are ms, median over {len(frames)} frames)")
