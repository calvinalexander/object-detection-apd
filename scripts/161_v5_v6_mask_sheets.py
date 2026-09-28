"""Sheets for the events where v5 and v6 disagree about the mask.

v6 was trained with 139 verified chin-worn NEGATIVES; on a held-out test it
separates chin (p 0.13) from covering (p 0.83) while v5 gave both 0.97. But v6
scores worse against the event-level curation. Only item-level judgement on this
camera can settle it, so this renders the disagreements for review.

Two pools, balanced:
  v6_says_violation  v6 p_mask < 0.4 while v5 p_mask > 0.6   (v6 flags, v5 clears)
  v5_says_violation  v5 p_mask < 0.4 while v6 p_mask > 0.6   (v5 flags, v6 clears)

Judge each as: covering / chin / bare / hidden.
    python scripts/161_v5_v6_mask_sheets.py [--n 30]
"""
import argparse, csv, json, random, sqlite3
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()
ARM = ROOT/"artifacts/replay16_person/prod_device"
OUT = ROOT/"artifacts/v5_v6_mask"; OUT.mkdir(exist_ok=True)
S5 = {r["event"]: r for r in json.load(open(ROOT/"artifacts/scores/v5_16sep_scores.json"))}
S6 = {r["event"]: r for r in json.load(open(ROOT/"artifacts/scores/v6_16sep_scores.json"))}
db = sqlite3.connect(f"file:{ARM}/apd_storage/apd_events.db?mode=ro", uri=True)
info = {e: (json.loads(b), ip) for e, b, ip in db.execute(
    "select event_id,person_bbox,image_path from events where timestamp like '2026-09-16%'")}

pools = {"v6_flags_v5_clears": [e for e in S6 if e in S5 and S6[e]["p"][1] < 0.4 and S5[e]["p"][1] > 0.6],
         "v5_flags_v6_clears": [e for e in S6 if e in S5 and S5[e]["p"][1] < 0.4 and S6[e]["p"][1] > 0.6]}
rng = random.Random(4)
for name, pool in pools.items():
    print(f"{name}: {len(pool)} events")
    sel = rng.sample(pool, min(a.n, len(pool)))
    tiles, index = [], []
    for n, e in enumerate(sel):
        bbox, ip = info[e]
        im = cv2.imread(str(ARM/ip))
        if im is None: continue
        x1, y1, x2, y2 = [int(v) for v in bbox]; h = y2-y1
        c = im[max(0, y1-int(.05*h)):y1+int(.5*h), max(0, x1-25):min(im.shape[1], x2+25)]
        if c.size == 0: continue
        s = min(330/c.shape[1], 330/c.shape[0])
        z = cv2.resize(c, (max(1, int(c.shape[1]*s)), max(1, int(c.shape[0]*s))), interpolation=cv2.INTER_CUBIC)
        t = np.full((330, 330, 3), 245, np.uint8); t[:z.shape[0], :z.shape[1]] = z
        t = cv2.copyMakeBorder(t, 26, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, f"#{n} v5 p={S5[e]['p'][1]:.2f}  v6 p={S6[e]['p'][1]:.2f}  h={h}",
                    (5, 18), 0, 0.45, (0, 0, 0), 1)
        tiles.append(t); index.append({"n": n, "event_id": e, "v5_p": round(S5[e]["p"][1], 3),
                                       "v6_p": round(S6[e]["p"][1], 3), "box_h": h, "verdict": ""})
    for s0 in range(0, len(tiles), 12):
        g = tiles[s0:s0+12]
        while len(g) < 12: g.append(np.full_like(tiles[0], 255))
        cv2.imwrite(str(OUT/f"{name}_sheet_{s0//12}.jpg"),
                    np.vstack([np.hstack(g[i:i+4]) for i in range(0, 12, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(OUT/f"{name}_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
    print(f"   -> {(len(tiles)+11)//12} sheets in {OUT}")
