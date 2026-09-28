"""Second-filter check on the device's own 16 Sep violation captures.

Production event -> its own snapshot + person_bbox -> our person crop (+18% pad,
letterbox 320) -> person_ppe_v3 (effnet_b0) with the deployed camera-calibrated
cuts. For every item production flagged, does our model still flag it?
"""
import collections, json, sqlite3, sys
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

ROOT = Path("/home/ubuntu/tryan/apd")
ARM = ROOT/"artifacts/replay16_person/prod_device"
B = ROOT/"deploy/apd_person"
VER = sys.argv[2] if len(sys.argv) > 2 else "v3"
cfg = json.loads((B/f"apd_ours_person_ppe_{VER}.json").read_text())
CUTS = cfg["cuts"]; ITEMS = ["head_cap", "mask", "gloves"]
NAME = {"headcap": "head_cap", "mask": "mask", "gloves": "gloves"}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
sess = ort.InferenceSession(str(B/f"apd_ours_person_ppe_{VER}.onnx"), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name


def prep(c, s=320):
    h, w = c.shape[:2]; r = s/max(h, w)
    im = cv2.resize(c, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return ((cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)


db = sqlite3.connect(f"file:{ARM}/apd_storage/apd_events.db?mode=ro", uri=True)
rows = list(db.execute("select event_id,camera_id,person_bbox,violation,image_path from events "
                       "where timestamp like '2026-09-16%'"))
print(f"{len(rows)} device events on 16 Sep; "
      f"{sum(1 for r in rows if r[1] != '_not_recorded')} inside a replayed clip")

out = []
batch, meta = [], []


def flush():
    if not batch: return
    P = sess.run(None, {inp: np.stack(batch).astype(np.float32)})[0]   # ONNX already outputs probabilities
    for p, m in zip(P, meta):
        v = {}
        for k, it in enumerate(ITEMS):
            c = CUTS[it]; pp, pk = float(p[k]), float(p[3+k])
            strong = ("present_strong" in c and pp < c["present_strong"]
                      and pk >= c.get("known_floor", c["known_above"]))
            v[it] = ("violation" if strong else
                     "unknown" if pk < c["known_above"] else
                     "violation" if pp < c["present_below"] else "ok")
        out.append({**m, "ours": v, "p": [round(float(x), 3) for x in p]})
    batch.clear(); meta.clear()


miss = 0
for eid, cam, bbox, vio, ip in rows:
    sp = ARM/ip
    im = cv2.imread(str(sp))
    if im is None: miss += 1; continue
    H, W = im.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in json.loads(bbox)]
    pw, ph = x2-x1, y2-y1
    c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
    if c.size == 0 or min(c.shape[:2]) < 8: miss += 1; continue
    batch.append(prep(c)); meta.append({"event": eid, "clip": cam, "in_clip": cam != "_not_recorded",
                                        "vio": [NAME[x] for x in json.loads(vio)], "h": ph})
    if len(batch) == 64: flush()
flush()
print(f"{len(out)} events scored ({miss} unreadable)")
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/filter16.json", "w"))


def report(rs, title):
    n = collections.Counter(); kept = collections.Counter()
    for r in rs:
        for it in r["vio"]:
            n[it] += 1
            n["ALL"] += 1
            if r["ours"][it] == "violation": kept[it] += 1; kept["ALL"] += 1
            else: kept[(it, r["ours"][it])] += 1; kept[("ALL", r["ours"][it])] += 1
    print(f"\n== {title}: {len(rs)} events, {n['ALL']} flagged items")
    for it in ITEMS + ["ALL"]:
        if not n[it]: continue
        print(f"  {it:9} {kept[it]:5d}/{n[it]:<5d} kept  ({100*kept[it]/n[it]:4.1f}%)   "
              f"dropped: unknown {kept[(it,'unknown')]}, ok {kept[(it,'ok')]}")


report([r for r in out if r["in_clip"]], "events inside the replayed 16 Sep clips")
report(out, "all 16 Sep device events")
