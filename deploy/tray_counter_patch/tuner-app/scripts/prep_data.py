"""Export the 4x-speed s14 detections and the offline reference tracks to public/s14_x4_data.json.
Reference tracks come from the 1x detections (dets15.json) with frame numbers divided by 4; the detections are
the stand-in detector's output on s14_x4.mp4 (dets_x4.json, from s14/detect_clip.py).
Usage: python scripts/prep_data.py <s14 folder with dets15.json and dets_x4.json>   (run from tuner-app/)"""
import json, numpy as np, sys
from scipy.optimize import linear_sum_assignment
S = sys.argv[1]
D = json.load(open(f"{S}/dets15.json"))
bc = lambda d: np.array([(d[0] + d[2]) / 2, d[3]])
tracks, done, nid = {}, [], 0
for k, dets in enumerate(D):
    ids = list(tracks); P = [tracks[i]["pts"][-1] + tracks[i]["v"] for i in ids]; C = [bc(d) for d in dets]
    used_t, used_d = set(), set()
    if ids and C:
        M = np.array([[np.hypot(*(p - c)) for c in C] for p in P]); r, c = linear_sum_assignment(M)
        for a, b in zip(r, c):
            if M[a, b] < 45:
                t = tracks[ids[a]]; v = C[b] - t["pts"][-1]
                t["v"] = 0.6 * t["v"] + 0.4 * v; t["pts"].append(C[b]); t["k"].append(k); t["fill"].append(dets[b][6]); t["miss"] = 0
                used_t.add(ids[a]); used_d.add(b)
    for i in ids:
        if i not in used_t:
            tracks[i]["miss"] += 1
            if tracks[i]["miss"] > 8: done.append(tracks.pop(i))
    for b, c in enumerate(C):
        if b not in used_d: tracks[nid] = dict(pts=[c], k=[k], v=np.zeros(2), fill=[dets[b][6]], miss=0); nid += 1
done += tracks.values()
ref = [dict(k=[k / 4 for k in t["k"]], p=[round(float(v), 1) for pt in t["pts"] for v in pt], fill=round(float(np.median(t["fill"])), 3))
       for t in done if len(t["k"]) > 1]
X = json.load(open(f"{S}/dets_x4.json"))
dets = [[d[:4] + [d[5]] for d in fr] for fr in X]
json.dump(dict(fps=15, w=1280, h=720, speed=4, dets=dets, ref=ref), open("public/s14_x4_data.json", "w"), separators=(",", ":"))
print(len(dets), "frames", len(ref), "ref tracks")
