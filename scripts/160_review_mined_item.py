"""Review sheets for mined positives of ANY item that v5 disagrees with.

Production's `mask: ok` means "a mask is on this person", not "covering nose and
mouth": a sample of 24 mined positives held ~5 chin-worn masks. So each row has
to be judged before it can train anything.

    python scripts/157_review_mined_mask.py [--mode disagree|agree] [--start 0] [--n 120]
Writes artifacts/mined_mask_audit/<mode>/sheet_NNN.jpg + index.csv
Verdicts go in artifacts/mined_mask_audit/verdicts.csv:
    covering  -> keep as a positive (mask present)
    chin      -> flip to a NEGATIVE (mask absent) - the most valuable rows
    hidden    -> drop (face not visible; neither label is supportable)
"""
import argparse, csv
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--item", default="mask", choices=["head_cap", "mask", "gloves"])
ap.add_argument("--mode", default="disagree", choices=["disagree", "agree"])
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--per", type=int, default=12)
a = ap.parse_args()
D = ROOT/"data/mined_positives"
OUT = ROOT/f"artifacts/mined_{a.item}_audit"/a.mode
OUT.mkdir(parents=True, exist_ok=True)
rows = [r for r in csv.DictReader(open(D/"candidates.csv")) if r["item"] == a.item]
K = {"head_cap": 0, "mask": 1, "gloves": 2}[a.item]
rows = [r for r in rows if (float(r["v5_p"]) < 0.5) == (a.mode == "disagree")]
rows.sort(key=lambda r: r["file"])
sel = rows[a.start:a.start+a.n]
print(f"{len(rows)} mask rows in mode {a.mode}; rendering {len(sel)} from {a.start}")

tiles, index = [], []
for n, r in enumerate(sel, start=a.start):
    im = cv2.imread(str(D/r["file"]))
    if im is None: continue
    top = im[:int(0.55*im.shape[0])] if a.item != "gloves" else im[int(0.2*im.shape[0]):int(0.95*im.shape[0])]
    s = min(330/top.shape[1], 330/top.shape[0])
    z = cv2.resize(top, (max(1, int(top.shape[1]*s)), max(1, int(top.shape[0]*s))), interpolation=cv2.INTER_CUBIC)
    t = np.full((330, 330, 3), 245, np.uint8); t[:z.shape[0], :z.shape[1]] = z
    t = cv2.copyMakeBorder(t, 26, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(t, f"#{n} {r['site']} p={r['v5_p']} h={r['box_h']}", (5, 18), 0, 0.45, (0, 0, 0), 1)
    tiles.append(t)
    index.append({"n": n, "file": r["file"], "site": r["site"], "event_id": r["event_id"],
                  "v5_p": r["v5_p"], "box_h": r["box_h"]})
for s0 in range(0, len(tiles), a.per):
    g = tiles[s0:s0+a.per]
    while len(g) < a.per: g.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(OUT/f"sheet_{(a.start+s0)//a.per:03d}.jpg"),
                np.vstack([np.hstack(g[i:i+4]) for i in range(0, a.per, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 95])
mode = "a" if (OUT/"index.csv").exists() and a.start else "w"
with open(OUT/"index.csv", mode, newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(index[0]))
    if mode == "w": w.writeheader()
    w.writerows(index)
print(f"{(len(tiles)+a.per-1)//a.per} sheets -> {OUT}")
