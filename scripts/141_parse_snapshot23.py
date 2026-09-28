"""Recover the device's person box and verdicts from data/snapshot/23 overlays.

These snapshots arrived without an events database, so everything the device
decided has to be read back from what it drew: a red person rectangle and a dark
panel listing headcap / gloves / mask, coloured green (ok), red (violation) or
white (unknown), plus a burned-in timestamp.

    python scripts/141_parse_snapshot23.py [--n 0] [--viz 12]
Writes data/snapshot/23_parsed.csv and artifacts/snapshot23/check.jpg
"""
import argparse, csv, glob, os
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--viz", type=int, default=12)
a = ap.parse_args()
ITEMS = ["headcap", "gloves", "mask"]


def panel_box(im):
    """The dark verdict panel, found by sliding a panel-sized window for the darkest patch."""
    dark = (im.max(axis=2) < 50).astype(np.float32)
    ii = cv2.integral(dark)
    H, W = dark.shape; bh, bw = 86, 190
    best = (-1.0, None)
    for y in range(0, H-bh, 4):
        for x in range(0, W-bw, 4):
            f = (ii[y+bh, x+bw] - ii[y, x+bw] - ii[y+bh, x] + ii[y, x])/(bh*bw)
            if f > best[0]: best = (f, (x, y, bw, bh))
    if best[0] < 0.55: return None
    x, y, w, h = best[1]
    col = dark[y:y+h].mean(axis=0); row = dark[:, x:x+w].mean(axis=1)
    gx = [i for i in range(W) if col[i] > 0.5 and abs(i-x) < 260]
    gy = [i for i in range(H) if row[i] > 0.5 and abs(i-y) < 140]
    if gx and gy: x, y, w, h = min(gx), min(gy), max(gx)-min(gx)+1, max(gy)-min(gy)+1
    return (x, y, min(w, 300), min(h, 120))


def person_box(im, panel):
    """Red rectangle outline, ignoring pixels inside the panel."""
    b, g, r = im[:, :, 0].astype(int), im[:, :, 1].astype(int), im[:, :, 2].astype(int)
    m = ((r > 110) & (r - g > 60) & (r - b > 60)).astype(np.uint8)
    if panel:
        x, y, w, h = panel; m[max(0, y-4):y+h+4, max(0, x-4):x+w+4] = 0
    m[:26, :] = 0                                        # burned-in timestamp row
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    best = None
    for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(c)
        if w < 12 or h < 25 or h/max(w, 1) > 8: continue
        edge = cv2.countNonZero(m[y:y+h, x:x+w])/max(1, 2*(w+h))   # outline, not a blob
        if 0.25 < edge < 4 and (best is None or w*h > best[2]*best[3]): best = (x, y, w, h)
    return best


def verdicts(im, panel):
    """Three text lines under the ID line; colour says the verdict."""
    if not panel: return {}
    x, y, w, h = panel
    roi = im[y:y+h, x:x+w]
    rows = np.where((roi.max(axis=2) > 90).any(axis=1))[0]        # rows holding text
    if len(rows) == 0: return {}
    runs, start = [], rows[0]
    for i in range(1, len(rows)):
        if rows[i] != rows[i-1]+1: runs.append((start, rows[i-1])); start = rows[i]
    runs.append((start, rows[-1]))
    keep = []
    for r0, r1 in runs:
        band = roi[r0:r1+1]
        px = band[band.max(axis=2) > 90].astype(int)
        if r1-r0 >= 4 and len(px) >= 80: keep.append((r0, r1, px))
    keep = keep[1:4]                                             # line 0 is the ID
    out = {}
    for (r0, r1, px), item in zip(keep, ITEMS):
        b, g, r = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
        out[item] = ("violation" if r - g > 40 else
                     "ok" if g - r > 40 else "unknown")           # yellow/white = cannot judge
    return out


files = sorted(glob.glob(str(ROOT/"data/snapshot/23/*.jpg")))
if a.n: files = files[:a.n]
rows, viz = [], []
for i, f in enumerate(files):
    im = cv2.imread(f)
    if im is None: continue
    p = panel_box(im); pb = person_box(im, p); v = verdicts(im, p)
    rows.append({"event_id": Path(f).stem, "mtime": int(os.path.getmtime(f)),
                 "box": "" if not pb else f"{pb[0]},{pb[1]},{pb[0]+pb[2]},{pb[1]+pb[3]}",
                 **{it: v.get(it, "") for it in ITEMS}})
    if len(viz) < a.viz and pb and len(v) == 3:
        o = im.copy(); cv2.rectangle(o, (pb[0], pb[1]), (pb[0]+pb[2], pb[1]+pb[3]), (255, 255, 0), 2)
        cv2.putText(o, " ".join(f"{k}:{x[:4]}" for k, x in v.items()), (4, 44), 0, 0.45, (255, 255, 0), 1)
        viz.append(cv2.resize(o, (352, 288)))
    if i % 500 == 0: print(f"  {i}/{len(files)}", flush=True)
out = ROOT/"data/snapshot/23_parsed.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import collections
print(f"{len(rows)} snapshots -> {out}")
print("  person box found:", sum(1 for r in rows if r["box"]))
for it in ITEMS:
    print(f"  {it:8}", dict(collections.Counter(r[it] for r in rows)))
if viz:
    (ROOT/"artifacts/snapshot23").mkdir(exist_ok=True)
    while len(viz) % 3: viz.append(np.zeros_like(viz[0]))
    cv2.imwrite(str(ROOT/"artifacts/snapshot23/check.jpg"),
                np.vstack([np.hstack(viz[i:i+3]) for i in range(0, len(viz), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 92])
