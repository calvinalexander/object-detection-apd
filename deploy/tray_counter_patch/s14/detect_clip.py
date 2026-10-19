"""Run the stand-in detector on every frame of a clip that is already at 15 fps; write detections JSON.
Usage: python detect_clip.py <video> <out.json>"""
import cv2, json, sys, time
from det import tray_masks
from fill import features
CUTS = [(0.04, 0), (0.17, 3), (0.27, 2), (0.32, 4)]          # same as detect_all.py; else class 1 (full)
def fill_class(fo):
    for c, k in CUTS:
        if fo < c: return k
    return 1
cap = cv2.VideoCapture(sys.argv[1]); out = []; t0 = time.time()
while True:
    ok, f = cap.read()
    if not ok: break
    out.append([[*map(int, b), 0.9, fill_class(fo), round(fo, 3), round(sz, 2)]
                for b, mm, a in tray_masks(f) for fo, sz in [features(f, b, mm)]])
json.dump(out, open(sys.argv[2], "w"))
print(len(out), "frames,", sum(map(len, out)), "detections,", f"{time.time() - t0:.0f}s")
