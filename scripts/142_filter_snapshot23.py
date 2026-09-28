"""Our v3 classifier as a second filter on the OTHER site's production alerts.

data/snapshot/23 holds 2,499 device snapshots with their own events database
(data/snapshot/23/db): camera "masak" on device ED-A69C5A-4396, 20-23 Sep --
a different kitchen from the 16/17 Sep footage (ED-97CE82-3753).

Same method as scripts/134: production's person box, cropped from production's
own snapshot (+18% pad, letterbox 320), re-judged by
deploy/apd_person/apd_ours_person_ppe_v3.onnx with the cuts in its JSON.

    python scripts/142_filter_snapshot23.py [out.json]
"""
import collections, json, sqlite3, sys
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT/"data/snapshot/23"
B = ROOT/"deploy/apd_person"
cfg = json.loads((B/"apd_ours_person_ppe_v3.json").read_text())
CUTS = cfg["cuts"]; ITEMS = ["head_cap", "mask", "gloves"]
NAME = {"headcap": "head_cap", "mask": "mask", "gloves": "gloves"}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
sess = ort.InferenceSession(str(B/"apd_ours_person_ppe_v3.onnx"), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name


def prep(c, s=320):
    h, w = c.shape[:2]; r = s/max(h, w)
    im = cv2.resize(c, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return ((cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)


db = sqlite3.connect(f"file:{SNAP}/db?mode=ro", uri=True)
rows = list(db.execute("select event_id,timestamp,person_bbox,violation from events"))
det = collections.defaultdict(dict)
for e, p, s in db.execute("select event_id,ppe_type,status from apd_detail"): det[e][NAME.get(p, p)] = s
print(f"{len(rows)} events in the database")

out, batch, meta, miss = [], [], [], 0


def flush():
    if not batch: return
    P = sess.run(None, {inp: np.stack(batch).astype(np.float32)})[0]
    for p, m in zip(P, meta):
        v = {}
        for k, it in enumerate(ITEMS):
            c = CUTS[it]; pp, pk = float(p[k]), float(p[3+k])
            strong = ("present_strong" in c and pp < c["present_strong"]
                      and pk >= c.get("known_floor", c["known_above"]))
            v[it] = ("violation" if strong else "unknown" if pk < c["known_above"]
                     else "violation" if pp < c["present_below"] else "ok")
        out.append({**m, "ours": v, "p": [round(float(x), 3) for x in p]})
    batch.clear(); meta.clear()


for eid, ts, bbox, vio in rows:
    f = SNAP/f"{eid}.jpg"
    im = cv2.imread(str(f)) if f.exists() else None
    if im is None: miss += 1; continue
    H, W = im.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in json.loads(bbox)]
    pw, ph = x2-x1, y2-y1
    c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
    if c.size == 0 or min(c.shape[:2]) < 8: miss += 1; continue
    batch.append(prep(c))
    meta.append({"event": eid, "time": ts, "vio": [NAME[x] for x in json.loads(vio)],
                 "detail": det.get(eid, {}), "h": ph})
    if len(batch) == 64: flush()
flush()
print(f"{len(out)} scored ({miss} without a usable snapshot)")
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/filter23.json", "w"))

n, kept = collections.Counter(), collections.Counter()
for r in out:
    for it in r["vio"]:
        n[it] += 1; n["ALL"] += 1
        if r["ours"][it] == "violation": kept[it] += 1; kept["ALL"] += 1
        else: kept[(it, r["ours"][it])] += 1; kept[("ALL", r["ours"][it])] += 1
print(f"\nproduction alerts re-judged ({n['ALL']} flagged items on {len(out)} events)")
for it in ITEMS + ["ALL"]:
    if not n[it]: continue
    print(f"  {it:9} {kept[it]:5d}/{n[it]:<5d} kept ({100*kept[it]/n[it]:4.1f}%)   "
          f"dropped: unknown {kept[(it,'unknown')]}, ok {kept[(it,'ok')]}")
ev = sum(1 for r in out if any(r["ours"][i] == "violation" for i in r["vio"]))
anyv = sum(1 for r in out if any(r["ours"][i] == "violation" for i in ITEMS))
print(f"events still an alert on production's item: {ev}/{len(out)}; on any item: {anyv}/{len(out)}")
extra = collections.Counter(i for r in out for i in ITEMS if i not in r["vio"] and r["ours"][i] == "violation")
print("violations we add that production did not raise:", dict(extra))
