"""Run the stand-in detector on every 2nd frame (30 -> 15 fps, like recorder.py) and cache the detections."""
import cv2, json, numpy as np, time
from det import tray_masks
from fill import features
V = "/home/ubuntu/tryan/apd/artifacts/tray_counter/data/s14_all_cases_elev45_az35_dist1700_hfov55.mp4"
CUTS = [(0.04, 0), (0.17, 3), (0.27, 2), (0.32, 4)]          # food score -> class; else 1 (full)
def fill_class(fo):
    for c, k in CUTS:
        if fo < c: return k
    return 1
cap = cv2.VideoCapture(V); out = []; i = 0; t0 = time.time()
while True:
    ok, f = cap.read()
    if not ok: break
    if i % 2 == 0:
        dets = []
        for b, mm, a in tray_masks(f):
            fo, sz = features(f, b, mm)
            dets.append([*map(int, b), 0.9, fill_class(fo), round(fo, 3), round(sz, 2)])
        out.append(dets)
    i += 1
json.dump(out, open("dets15.json", "w"))
print(len(out), "frames at 15 fps,", sum(map(len, out)), "detections,", f"{time.time() - t0:.0f}s")
