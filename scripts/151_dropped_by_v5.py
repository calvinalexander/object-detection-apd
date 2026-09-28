"""The alerts v3 raised and v5 drops: artifacts/dropped_v3_to_v5/.

953 events that sit in artifacts/filter16/flagged (v3) but in
artifacts/filter16_v5/passed (v5). Nobody has reviewed these -- the manual
curation only covered events v3 had passed -- so they are the open question
behind the day-level "38% fewer alerts".

Each image is the device snapshot with production's person box drawn and three
header lines: what the device flagged, what v3 said, what v5 says.

    python scripts/151_dropped_by_v5.py
"""
import csv, json, shutil, sqlite3
from pathlib import Path
import cv2, collections

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT/"artifacts/replay16_person/prod_device"
OUT = ROOT/"artifacts/dropped_v3_to_v5"
ITEMS = ["head_cap", "mask", "gloves"]
SH = {"head_cap": "cap", "mask": "mask", "gloves": "glv"}
COL = {"violation": (0, 0, 220), "ok": (0, 150, 0), "unknown": (150, 150, 150)}

v3 = {r["event_id"]: r for r in csv.DictReader(open(ROOT/"artifacts/filter16/index.csv"))}
v5 = {r["event_id"]: r for r in csv.DictReader(open(ROOT/"artifacts/filter16_v5/index.csv"))}
drop = [e for e in v5 if e in v3
        and (v3[e]["ours_kept"] or v3[e]["ours_extra"])
        and not (v5[e]["ours_kept"] or v5[e]["ours_extra"])]
print(f"{len(drop)} events flagged by v3 and passed by v5")

if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True)
db = sqlite3.connect(f"file:{ARM}/apd_storage/apd_events.db?mode=ro", uri=True)
info = {e: (b, ip, ts) for e, b, ip, ts in db.execute(
    "select event_id,person_bbox,image_path,timestamp from events where timestamp like '2026-09-16%'")}

idx, why = [], collections.Counter()
for e in sorted(drop, key=lambda e: v5[e]["time"]):
    bbox, ip, ts = info[e]
    im = cv2.imread(str(ARM/ip))
    if im is None: continue
    x1, y1, x2, y2 = [int(t) for t in json.loads(bbox)]
    cv2.rectangle(im, (x1, y1), (x2, y2), (200, 200, 0), 2)
    cv2.rectangle(im, (0, 0), (im.shape[1], 48), (30, 30, 30), -1)
    cv2.putText(im, "DEVICE: " + (v5[e]["device_violation"] or "-"), (6, 13), 0, 0.42, (255, 255, 255), 1)
    for row, (src, y) in enumerate(((v3, 28), (v5, 43))):
        x = 6
        cv2.putText(im, "v3:" if row == 0 else "v5:", (x, y), 0, 0.42, (255, 255, 255), 1); x += 26
        for i in ITEMS:
            t = f"{SH[i]} {src[e]['ours_'+i][:4]}"
            cv2.putText(im, t, (x, y), 0, 0.42, COL[src[e]["ours_"+i]], 1); x += 10*len(t)
    lost = [i for i in ITEMS if v3[e]["ours_"+i] == "violation" and v5[e]["ours_"+i] != "violation"]
    for i in lost: why[f"{i}: {v5[e]['ours_'+i]}"] += 1
    tag = "-".join(SH[i] for i in lost) or "none"
    name = ts.replace(" ", "_").replace(":", "-") + f"_was-{tag}_" + e[:8] + ".jpg"
    cv2.imwrite(str(OUT/name), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    idx.append({"file": name, "event_id": e, "time": ts,
                "device_violation": v5[e]["device_violation"],
                "v3_flagged": "+".join(x for x in (v3[e]["ours_kept"], v3[e]["ours_extra"]) if x),
                "lost_items": "+".join(lost), "curation": v5[e]["curation"],
                **{f"v5_{i}": v5[e]["ours_"+i] for i in ITEMS},
                **{f"p_{i}": v5[e]["p_"+i] for i in ITEMS},
                **{f"known_{i}": v5[e]["known_"+i] for i in ITEMS}})
with open(OUT/"index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(idx[0])); w.writeheader(); w.writerows(idx)
print(f"wrote {len(idx)} images to {OUT}")
print("why the alert was lost:", dict(why))
print("already curated among them:", collections.Counter(r["curation"] for r in idx if r["curation"]))
