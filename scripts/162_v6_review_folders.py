"""Review folders for v6 on 16 Sep, split by what changed and what went wrong.

artifacts/v6_review/
  flagged_to_passed/   v3 (deployed cuts) flagged it, v6 passes it   -- the alerts v6 drops
  false_alarms/<item>/ your curation says skip, v6 flags it          -- by the item that fired
  still_missed/        your curation says flag, v6 passes it
  index.csv            one row per image, both verdicts and all six scores

v6 cuts are its own val fit: head_cap 0.80/0.30, mask 0.45/0.575, gloves 0.80/0.575.

    python scripts/162_v6_review_folders.py
"""
import csv, json, re, shutil, sqlite3, collections
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT/"artifacts/replay16_person/prod_device"
OUT = ROOT/"artifacts/v6_review"
ITEMS = ["head_cap", "mask", "gloves"]
SH = {"head_cap": "cap", "mask": "mask", "gloves": "glv"}
COL = {"violation": (0, 0, 220), "ok": (0, 150, 0), "unknown": (150, 150, 150)}
V6 = {"head_cap": (0.80, 0.30), "mask": (0.45, 0.575), "gloves": (0.80, 0.575)}
V3 = {"head_cap": (0.87, 0.35), "mask": (0.98, 0.55), "gloves": (0.969, 0.475)}

S6 = {r["event"]: r for r in json.load(open(ROOT/"artifacts/scores/v6_16sep_scores.json"))}
S3 = {r["event"]: r for r in json.load(open(ROOT/"artifacts/scores/v3_16sep_scores.json"))}
cur = {}
for r in csv.DictReader(open(ROOT/"artifacts/_ss/passed_filenames.csv")):
    if r["Status"].strip() in ("skip", "flag"):
        cur[re.sub(r"\.jpg$", "", r["Filename"].strip()).split("_")[-1]] = r["Status"].strip()
db = sqlite3.connect(f"file:{ARM}/apd_storage/apd_events.db?mode=ro", uri=True)
info = {e: (json.loads(b), ip, ts, v) for e, b, ip, ts, v in db.execute(
    "select event_id,person_bbox,image_path,timestamp,violation from events where timestamp like '2026-09-16%'")}

if OUT.exists(): shutil.rmtree(OUT)
for d in ["flagged_to_passed", "still_missed"] + [f"false_alarms/{i}" for i in ITEMS]:
    (OUT/d).mkdir(parents=True)


def verd(p, C):
    return {i: ("unknown" if p[3+k] < C[i][0] else ("violation" if p[k] < C[i][1] else "ok"))
            for k, i in enumerate(ITEMS)}


idx, n = [], collections.Counter()
for e, r in S6.items():
    if e not in info: continue
    v6 = verd(r["p"], V6); v3 = verd(S3[e]["p"], V3) if e in S3 else None
    f6 = any(x == "violation" for x in v6.values())
    f3 = v3 and any(x == "violation" for x in v3.values())
    c = cur.get(e[:8], "")
    fired = [i for i in ITEMS if v6[i] == "violation"]
    dests = []
    if f3 and not f6: dests.append("flagged_to_passed")
    if c == "skip" and f6: dests += [f"false_alarms/{i}" for i in fired]
    if c == "flag" and not f6: dests.append("still_missed")
    if not dests: continue
    bbox, ip, ts, vio = info[e]
    im = cv2.imread(str(ARM/ip))
    if im is None: continue
    x1, y1, x2, y2 = [int(t) for t in bbox]
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255) if f6 else (200, 200, 0), 2)
    cv2.rectangle(im, (0, 0), (im.shape[1], 48), (30, 30, 30), -1)
    cv2.putText(im, f"DEVICE: {','.join(json.loads(vio))}" + (f"   YOUR CURATION: {c}" if c else ""),
                (6, 13), 0, 0.42, (255, 255, 255), 1)
    for row, (lab, vv) in enumerate((("v3", v3), ("v6", v6))):
        if vv is None: continue
        x = 6; y = 28 + row*15
        cv2.putText(im, lab + ":", (x, y), 0, 0.42, (255, 255, 255), 1); x += 26
        for i in ITEMS:
            t = f"{SH[i]} {vv[i][:4]}"
            cv2.putText(im, t, (x, y), 0, 0.42, COL[vv[i]], 1); x += 10*len(t)
    name = ts.replace(" ", "_").replace(":", "-") + "_" + ("-".join(SH[i] for i in fired) or "clear") + f"_{e[:8]}.jpg"
    for d in dests:
        cv2.imwrite(str(OUT/d/name), im, [cv2.IMWRITE_JPEG_QUALITY, 85]); n[d] += 1
    idx.append({"file": f"{dests[0]}/{name}", "event_id": e, "time": ts, "curation": c,
                "device": "+".join(json.loads(vio)),
                "v3": "+".join(i for i in ITEMS if v3 and v3[i] == "violation"),
                "v6": "+".join(fired), "folders": ";".join(dests),
                **{f"v6_{i}": v6[i] for i in ITEMS},
                **{f"p_{i}": round(r["p"][k], 3) for k, i in enumerate(ITEMS)},
                **{f"k_{i}": round(r["p"][3+k], 3) for k, i in enumerate(ITEMS)}})

with open(OUT/"index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(idx[0])); w.writeheader(); w.writerows(idx)
for k in sorted(n): print(f"  {k:26} {n[k]}")
print(f"index rows {len(idx)} -> {OUT}/index.csv")
