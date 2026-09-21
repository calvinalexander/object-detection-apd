"""Split production's 16 Sep violation captures by what our model says about them.

    artifacts/filter16/flagged/   our model calls at least one item a violation -- whether or
                                  not that is the item production flagged
    artifacts/filter16/passed/    our model finds nothing wrong on this person
    artifacts/filter16/index.csv  one row per event

Each image is the device's own snapshot with the device person box drawn, the
device verdict on the left and ours on the right. Files from clips that were
replayed carry a CLIP_ prefix.
"""
import csv, json, shutil, sqlite3
from pathlib import Path
import cv2

ROOT = Path("/home/ubuntu/tryan/apd")
ARM = ROOT/"artifacts/replay16_person/prod_device"
OUT = ROOT/"artifacts/filter16"
ITEMS = ["head_cap", "mask", "gloves"]
SH = {"head_cap": "cap", "mask": "mask", "gloves": "glv"}
COL = {"violation": (0, 0, 220), "ok": (0, 150, 0), "unknown": (150, 150, 150)}

res = {r["event"]: r for r in json.load(open("/home/ubuntu/.claude/jobs/ba1805c7/tmp/filter16.json"))}
db = sqlite3.connect(f"file:{ARM}/apd_storage/apd_events.db?mode=ro", uri=True)
rows = list(db.execute("select event_id,timestamp,camera_id,person_bbox,image_path from events "
                       "where timestamp like '2026-09-16%'"))
for d in ("flagged", "passed"):
    if (OUT/d).exists(): shutil.rmtree(OUT/d)
    (OUT/d).mkdir(parents=True)

idx = []
for eid, ts, cam, bbox, ip in rows:
    r = res.get(eid)
    if r is None: continue
    kept = [i for i in r["vio"] if r["ours"][i] == "violation"]
    extra = [i for i in ITEMS if i not in r["vio"] and r["ours"][i] == "violation"]
    dest = "flagged" if kept or extra else "passed"
    im = cv2.imread(str(ARM/ip))
    if im is None: continue
    x1, y1, x2, y2 = [int(v) for v in json.loads(bbox)]
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255) if kept or extra else (200, 200, 0), 2)
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(im, "DEVICE: " + ",".join(SH[i] for i in r["vio"]), (6, 14), 0, 0.45, (255, 255, 255), 1)
    x = 6
    cv2.putText(im, "OURS:", (x, 29), 0, 0.45, (255, 255, 255), 1); x += 44
    for i in ITEMS:
        t = f"{SH[i]} {r['ours'][i][:4]}"
        cv2.putText(im, t, (x, 29), 0, 0.45, COL[r["ours"][i]], 1); x += 11*len(t)
    tag = "-".join(SH[i] for i in kept + extra) or "clear"
    name = ("CLIP_" if r["in_clip"] else "") + ts.replace(" ", "_").replace(":", "-") + f"_{tag}_" + eid[:8] + ".jpg"
    cv2.imwrite(str(OUT/dest/name), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    idx.append({"file": f"{dest}/{name}", "event_id": eid, "time": ts, "clip": cam,
                "in_replayed_clip": int(r["in_clip"]), "device_violation": "+".join(r["vio"]),
                "ours_kept": "+".join(kept), "ours_extra": "+".join(extra),
                **{f"ours_{i}": r["ours"][i] for i in ITEMS},
                **{f"p_{i}": r["p"][k] for k, i in enumerate(ITEMS)},
                **{f"known_{i}": r["p"][3+k] for k, i in enumerate(ITEMS)}})

with open(OUT/"index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(idx[0])); w.writeheader(); w.writerows(idx)
f = sum(1 for r in idx if r["ours_kept"] or r["ours_extra"]); c = sum(1 for r in idx if r["in_replayed_clip"])
print(f"flagged {f}  passed {len(idx)-f}  (of {len(idx)} events; {c} from replayed clips)")
