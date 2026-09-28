"""Sample sheets for the other site (data/snapshot/23), as the model sees them.

Each tile is production's person box cropped and letterboxed to 320 -- exactly
the classifier's input -- with the device's own per-item verdict and v5's score
and verdict beside it.

    python scripts/153_sample_sheet_23.py [random|disagree] [--cuts curation|val|hybrid]
"""
import argparse, json, random, sqlite3
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("mode", nargs="?", default="random", choices=["random", "disagree"])
ap.add_argument("--cuts", default="curation", choices=["curation", "val", "hybrid", "deployed", "median"])
ap.add_argument("--scores", default="artifacts/scores/v5_23sep_scores.json")
ap.add_argument("--tag", default="v5")
a = ap.parse_args()
SNAP = ROOT/"data/snapshot/23"
ITEMS = ["head_cap", "mask", "gloves"]
CUTS = {"curation": {"head_cap": (0.95, 0.05), "mask": (0.10, 0.275), "gloves": (0.80, 0.10)},
        "val":      {"head_cap": (0.45, 0.325), "mask": (0.55, 0.575), "gloves": (0.45, 0.275)},
        "hybrid":   {"head_cap": (0.45, 0.325), "mask": (0.10, 0.275), "gloves": (0.80, 0.10)},
        "deployed": {"head_cap": (0.87, 0.35), "mask": (0.98, 0.55), "gloves": (0.969, 0.475)},
        "median":   None}[a.cuts]
S = {r["event"]: r for r in json.load(open(ROOT/a.scores))}
if CUTS is None:                      # known_above = this camera's own median score per item
    import numpy as _np
    PB = {"v5": {"head_cap": 0.325, "mask": 0.575, "gloves": 0.275},   # each model's own val-fitted presence cuts
          "v3": {"head_cap": 0.350, "mask": 0.550, "gloves": 0.475}}[a.tag]
    CUTS = {i: (float(_np.median([r["p"][3+k] for r in S.values()])), PB[i]) for k, i in enumerate(ITEMS)}
    print("median cuts:", {i: (round(CUTS[i][0], 3), CUTS[i][1]) for i in ITEMS})
db = sqlite3.connect(f"file:{SNAP}/db?mode=ro", uri=True)
box = {e: json.loads(b) for e, b in db.execute("select event_id,person_bbox from events")}


def verdict(p):
    out = {}
    for k, i in enumerate(ITEMS):
        ka, pb = CUTS[i]
        out[i] = "unknown" if p[3+k] < ka else ("violation" if p[k] < pb else "ok")
    return out


rows = [S[e] for e in sorted(S)]
if a.mode == "disagree":
    rows = [r for r in rows if any((r["detail"].get(i) == "violation") != (verdict(r["p"])[i] == "violation")
                                   for i in ITEMS if i in r["detail"])]
print(f"{len(rows)} events in the {a.mode} pool (cuts: {a.cuts})")
sel = random.Random(5).sample(rows, min(12, len(rows)))


def lb(im, s=320):
    h, w = im.shape[:2]; r = s/max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return o


COL = {"violation": (0, 0, 200), "ok": (0, 120, 0), "unknown": (130, 130, 130)}
tiles = []
for r in sel:
    im = cv2.imread(str(SNAP/f"{r['event']}.jpg"))
    if im is None: continue
    H, W = im.shape[:2]; x1, y1, x2, y2 = [int(v) for v in box[r["event"]]]
    pw, ph = x2-x1, y2-y1
    c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
    if c.size == 0: continue
    v = verdict(r["p"])
    t = cv2.copyMakeBorder(cv2.resize(lb(c), (330, 330)), 56, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(t, f"{r['time'][5:]}  box {pw}x{ph}", (5, 13), 0, 0.38, (110, 110, 110), 1)
    y = 28
    for k, i in enumerate(ITEMS):
        dev = r["detail"].get(i, "-")
        bad = i in r["detail"] and (dev == "violation") != (v[i] == "violation")
        cv2.putText(t, f"{i[:3]} device {dev:9} {a.tag} p={r['p'][k]:.2f} k={r['p'][3+k]:.2f} -> {v[i]}",
                    (5, y), 0, 0.36, (0, 0, 200) if bad else COL[v[i]], 1)
        y += 13
    tiles.append(t)
while len(tiles) % 4: tiles.append(np.full_like(tiles[0], 255))
out = ROOT/f"artifacts/snapshot23_{a.mode}_{a.tag}_{a.cuts}.jpg"
cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 94])
print(out)
