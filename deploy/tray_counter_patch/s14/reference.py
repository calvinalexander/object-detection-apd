"""Independent reference: offline tracks over the clean stand-in boxes (class-agnostic, Hungarian on predicted
bottom-centre), crossing = bottom-centre changes side of the line and stays there >= 4 frames."""
import json, numpy as np
from scipy.optimize import linear_sum_assignment
D = json.load(open("dets15.json")); R = json.load(open("s14_events.json")); (x1, y1), (x2, y2) = R["line"]
L = np.hypot(x2 - x1, y2 - y1)
sd = lambda p: ((x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)) / L
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
ref = []
for t in done:
    s = np.sign([sd(p) for p in t["pts"]]); state = None; run = 0
    for j in range(len(s)):
        if s[j] == 0: continue
        if state is None: state = s[j]; continue
        if s[j] != state:
            if all(s[j:j + 4] == s[j]) and len(s[j:j + 4]) == 4:
                ref.append(dict(t=round(t["k"][j] / 15, 2), dir="out" if s[j] > 0 else "in",
                                fill=float(np.median(t["fill"]))))
                state = s[j]
ref.sort(key=lambda e: e["t"])
json.dump(ref, open("s14_reference.json", "w"), indent=1)
print("reference crossings:", len(ref), " OUT:", sum(e["dir"] == "out" for e in ref), " IN:", sum(e["dir"] == "in" for e in ref))
print("IN:", [e["t"] for e in ref if e["dir"] == "in"])
# match reference against each counter version
for v in ("production", "patched"):
    E = [dict(e) for e in R[v]]; used = set(); ok = wrong = miss = 0; bad = []
    for r in ref:
        j = next((j for j, e in enumerate(E) if j not in used and abs(e["t"] - r["t"]) < 1.5), None)
        if j is None: miss += 1; bad.append(("missed", r["t"], r["dir"])); continue
        used.add(j)
        if E[j]["dir"] == r["dir"]: ok += 1
        else: wrong += 1; bad.append(("wrong dir", r["t"], r["dir"]))
    extra = [(E[j]["t"], E[j]["dir"]) for j in range(len(E)) if j not in used]
    print(f"{v:10s} right={ok} wrong_dir={wrong} missed={miss} extra={len(extra)}  issues={bad}  extra={extra}")
