"""flagged/ and passed/ for 16 Sep using v5, in a folder of its own.

Scores: artifacts/scores/v5_16sep_scores.json (scripts/147)
Cuts:   fitted on the even hours of the manual curation, scored on the odd hours
        (scripts/148) -- held-out result 140/219 catches at 78/444 false alarms.

artifacts/filter16/ (v3) is never touched, so the two builds can be compared and
the curation Status column still matches by event id.

    python scripts/150_build_filter16_v5.py
"""
import csv, json, re, shutil, sqlite3
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT/"artifacts/replay16_person/prod_device"
OUT = ROOT/"artifacts/filter16_v5"
ITEMS = ["head_cap", "mask", "gloves"]
SH = {"head_cap": "cap", "mask": "mask", "gloves": "glv"}
COL = {"violation": (0, 0, 220), "ok": (0, 150, 0), "unknown": (150, 150, 150)}
CUTS = {"head_cap": {"known_above": 0.95, "present_below": 0.05},
        "mask": {"known_above": 0.10, "present_below": 0.275},
        "gloves": {"known_above": 0.80, "present_below": 0.10}}

sc = {r["event"]: r for r in json.load(open(ROOT/"artifacts/scores/v5_16sep_scores.json"))}
prev = {}
for r in csv.DictReader(open(ROOT/"artifacts/_ss/passed_filenames.csv")):
    if r["Status"].strip() in ("skip", "flag"):
        prev[re.sub(r"\.jpg$", "", r["Filename"].strip()).split("_")[-1]] = r["Status"].strip()

db = sqlite3.connect(f"file:{ARM}/apd_storage/apd_events.db?mode=ro", uri=True)
rows = list(db.execute("select event_id,timestamp,person_bbox,violation,image_path from events "
                       "where timestamp like '2026-09-16%'"))
for d in ("flagged", "passed"):
    if (OUT/d).exists(): shutil.rmtree(OUT/d)
    (OUT/d).mkdir(parents=True)

idx = []
for eid, ts, bbox, vio, ip in rows:
    r = sc.get(eid)
    if r is None: continue
    v = {}
    for k, it in enumerate(ITEMS):
        c = CUTS[it]; pp, pk = r["p"][k], r["p"][3+k]
        v[it] = "unknown" if pk < c["known_above"] else ("violation" if pp < c["present_below"] else "ok")
    kept = [i for i in r["vio"] if v[i] == "violation"]
    extra = [i for i in ITEMS if i not in r["vio"] and v[i] == "violation"]
    dest = "flagged" if kept or extra else "passed"
    im = cv2.imread(str(ARM/ip))
    if im is None: continue
    x1, y1, x2, y2 = [int(t) for t in json.loads(bbox)]
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255) if dest == "flagged" else (200, 200, 0), 2)
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(im, "DEVICE: " + ",".join(SH[i] for i in r["vio"]), (6, 14), 0, 0.45, (255, 255, 255), 1)
    x = 6
    cv2.putText(im, "V5:", (x, 29), 0, 0.45, (255, 255, 255), 1); x += 30
    for i in ITEMS:
        t = f"{SH[i]} {v[i][:4]}"
        cv2.putText(im, t, (x, 29), 0, 0.45, COL[v[i]], 1); x += 11*len(t)
    tag = "-".join(SH[i] for i in kept + extra) or "clear"
    name = ts.replace(" ", "_").replace(":", "-") + f"_{tag}_" + eid[:8] + ".jpg"
    cv2.imwrite(str(OUT/dest/name), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    idx.append({"file": f"{dest}/{name}", "event_id": eid, "time": ts,
                "device_violation": "+".join(r["vio"]), "ours_kept": "+".join(kept),
                "ours_extra": "+".join(extra), "curation": prev.get(eid[:8], ""),
                **{f"ours_{i}": v[i] for i in ITEMS},
                **{f"p_{i}": r["p"][k] for k, i in enumerate(ITEMS)},
                **{f"known_{i}": r["p"][3+k] for k, i in enumerate(ITEMS)}})

with open(OUT/"index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(idx[0])); w.writeheader(); w.writerows(idx)
f = sum(1 for r in idx if r["ours_kept"] or r["ours_extra"])
import collections
cur = collections.Counter((r["curation"], "flagged" if r["ours_kept"] or r["ours_extra"] else "passed")
                          for r in idx if r["curation"])
print(f"flagged {f}  passed {len(idx)-f}  (of {len(idx)} events)")
print("against your curation:", {f"{k[0]}->{k[1]}": v for k, v in sorted(cur.items())})
