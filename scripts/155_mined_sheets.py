"""Review sheets for the mined weak positives (scripts/154).

    python scripts/155_mined_sheets.py <item> [--mode disagree|agree|random] [--n 12]
"""
import argparse, csv, random
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("item", choices=["head_cap", "mask", "gloves"])
ap.add_argument("--mode", default="disagree", choices=["disagree", "agree", "random"])
ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
D = ROOT/"data/mined_positives"
rows = [r for r in csv.DictReader(open(D/"candidates.csv")) if r["item"] == a.item]
if a.mode == "disagree": rows = [r for r in rows if float(r["v5_p"]) < 0.5]
elif a.mode == "agree":  rows = [r for r in rows if float(r["v5_p"]) >= 0.5]
print(f"{len(rows)} {a.item} rows in mode {a.mode}")
sel = random.Random(7).sample(rows, min(a.n, len(rows)))
TH = TW = 360
tiles = []
for r in sel:
    im = cv2.imread(str(D/r["file"]))
    if im is None: continue
    top = im[:int(0.45*im.shape[0])] if a.item != "gloves" else im[int(0.25*im.shape[0]):int(0.9*im.shape[0])]
    s = min(TW/top.shape[1], TH/top.shape[0])
    z = cv2.resize(top, (max(1, int(top.shape[1]*s)), max(1, int(top.shape[0]*s))), interpolation=cv2.INTER_CUBIC)
    t = np.full((TH, TW, 3), 255, np.uint8); t[:z.shape[0], :z.shape[1]] = z
    t = cv2.copyMakeBorder(t, 30, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(t, f"{r['site']} device OK  v5 p={r['v5_p']} k={r['v5_k']}  h={r['box_h']}",
                (5, 20), 0, 0.45, (0, 0, 180) if float(r["v5_p"]) < 0.5 else (0, 120, 0), 1)
    tiles.append(t)
while len(tiles) % 4: tiles.append(np.full_like(tiles[0], 255))
out = ROOT/f"artifacts/mined_{a.item}_{a.mode}.jpg"
cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 94])
print(out)
