"""flagged/ and passed/ folders for the other site, mirroring artifacts/filter16.

    artifacts/filter23/flagged/   we call at least one item a violation
    artifacts/filter23/passed/    we find nothing wrong on this person
    artifacts/filter23/index.csv  one row per event

Each image is the device's own snapshot with its person box drawn, the device
verdict on the first header line and ours on the second.

    python scripts/143_filter23_folders.py [scores.json]
"""
import csv, json, shutil, sqlite3, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT/"data/snapshot/23"
OUT = ROOT/"artifacts/filter23"
ITEMS = ["head_cap", "mask", "gloves"]
SH = {"head_cap": "cap", "mask": "mask", "gloves": "glv"}
COL = {"violation": (0, 0, 220), "ok": (0, 150, 0), "unknown": (150, 150, 150)}

res = {r["event"]: r for r in json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/filter23.json"))}
db = sqlite3.connect(f"file:{SNAP}/db?mode=ro", uri=True)
rows = list(db.execute("select event_id,timestamp,person_bbox,violation from events"))
for d in ("flagged", "passed"):
    if (OUT/d).exists(): shutil.rmtree(OUT/d)
    (OUT/d).mkdir(parents=True)

idx = []
for eid, ts, bbox, vio in rows:
    r = res.get(eid)
    if r is None: continue
    kept = [i for i in r["vio"] if r["ours"][i] == "violation"]
    extra = [i for i in ITEMS if i not in r["vio"] and r["ours"][i] == "violation"]
    dest = "flagged" if kept or extra else "passed"
    im = cv2.imread(str(SNAP/f"{eid}.jpg"))
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
    name = ts.replace(" ", "_").replace(":", "-") + f"_{tag}_" + eid[:8] + ".jpg"
    cv2.imwrite(str(OUT/dest/name), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    idx.append({"file": f"{dest}/{name}", "event_id": eid, "time": ts,
                "device_violation": "+".join(r["vio"]), "ours_kept": "+".join(kept),
                "ours_extra": "+".join(extra),
                **{f"device_{i}": r["detail"].get(i, "") for i in ITEMS},
                **{f"ours_{i}": r["ours"][i] for i in ITEMS},
                **{f"p_{i}": r["p"][k] for k, i in enumerate(ITEMS)},
                **{f"known_{i}": r["p"][3+k] for k, i in enumerate(ITEMS)}})

with open(OUT/"index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(idx[0])); w.writeheader(); w.writerows(idx)
f = sum(1 for r in idx if r["ours_kept"] or r["ours_extra"])
print(f"flagged {f}  passed {len(idx)-f}  (of {len(idx)} events)")
