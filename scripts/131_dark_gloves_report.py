"""Where do the DARK (black/navy) glove examples in the training data come from?

Scans every gloves box in data/clean/det7, measures its median brightness (HSV V),
and writes the dark ones out for inspection with their source and size.

    python scripts/131_dark_gloves_report.py
Writes artifacts/dark_gloves/{boxes.csv, sheet_NN.jpg, README.md}
"""
import collections, csv, json
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/"data/clean/det7"; OUT = ROOT/"artifacts/dark_gloves"; OUT.mkdir(parents=True, exist_ok=True)
sites = {p.stem: p.parent.name for sd in (ROOT/"data/sites").iterdir() if sd.is_dir() for p in sd.iterdir()}
DARK = 70

rows, tiles = [], []
for sp in ("train", "val"):
    for lp in sorted((D/"labels"/sp).glob("*.txt")):
        ls = [l.split() for l in lp.read_text().splitlines() if l.strip()]
        if not any(int(f[0]) == 1 for f in ls): continue
        im = cv2.imread(str(D/"images"/sp/(lp.stem + ".jpg")))
        if im is None: continue
        H, W = im.shape[:2]
        frame = lp.stem.rsplit("_p", 1)[0]
        src = f"site {sites[frame]}" if frame in sites else "pre-annotated"
        for i, f in enumerate(ls):
            if int(f[0]) != 1: continue
            x, y, w, h = [float(v) for v in f[1:5]]
            x1, y1, x2, y2 = int((x-w/2)*W), int((y-h/2)*H), int((x+w/2)*W), int((y+h/2)*H)
            c = im[max(0, y1):y2, max(0, x1):x2]
            if c.size == 0 or min(c.shape[:2]) < 5: continue
            v = float(np.median(cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2]))
            rows.append({"split": sp, "image": lp.stem, "box": i, "source": src, "person_h_px": H,
                         "glove_w_px": x2-x1, "glove_h_px": y2-y1, "brightness": round(v, 1),
                         "dark": int(v < DARK)})
            if v < DARK:
                z = cv2.resize(c, (120, 120), interpolation=cv2.INTER_LANCZOS4)
                ctx = im[max(0, y1-2*(y2-y1)):y2+(y2-y1), max(0, x1-2*(x2-x1)):x2+2*(x2-x1)]
                ctx = cv2.resize(ctx, (120, 120)) if ctx.size else np.full((120, 120, 3), 235, np.uint8)
                t = cv2.copyMakeBorder(np.hstack([z, ctx]), 20, 4, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(t, f"{src[:14]} {x2-x1}px V{v:.0f}", (3, 14), 0, 0.35, (0, 0, 0), 1)
                tiles.append(t)
with open(OUT/"boxes.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
PER = 40
for s in range(0, len(tiles), PER):
    g = tiles[s:s+PER]; g += [np.full_like(tiles[0], 255)]*(PER-len(g))
    cv2.imwrite(str(OUT/f"sheet_{s//PER:02d}.jpg"), np.vstack([np.hstack(g[i:i+8]) for i in range(0, PER, 8)]),
                [cv2.IMWRITE_JPEG_QUALITY, 93])
dark = [r for r in rows if r["dark"]]
bysrc = collections.Counter(r["source"] for r in dark)
allsrc = collections.Counter(r["source"] for r in rows)
med = lambda a: float(np.median(a)) if a else 0
(OUT/"README.md").write_text(f"""# Dark glove examples in the training data (det7)

Every `gloves` box in data/clean/det7, classified by median brightness (HSV V).
"Dark" means V < {DARK}, i.e. black or very dark navy gloves.

| | boxes | median glove width | median person height |
|---|---|---|---|
| all gloves boxes | {len(rows)} | {med([r['glove_w_px'] for r in rows]):.0f} px | {med([r['person_h_px'] for r in rows]):.0f} px |
| dark ones | **{len(dark)}** ({len(dark)/len(rows):.0%}) | {med([r['glove_w_px'] for r in dark]):.0f} px | {med([r['person_h_px'] for r in dark]):.0f} px |

On the device camera the alerted people are ~118 px tall, so its gloves are far
smaller than these examples.

Dark gloves by source ({len(bysrc)} distinct sources):

| source | dark boxes | all gloves boxes there |
|---|---|---|
""" + "\n".join(f"| {k} | {v} | {allsrc[k]} |" for k, v in bysrc.most_common()) + f"""

Sheets: each tile is the glove box (left) and its surroundings (right), with
source, box width and brightness. boxes.csv lists every gloves box with its
measurements, so the cut can be re-drawn at a different brightness.
""")
print(f"{len(rows)} gloves boxes, {len(dark)} dark ({len(dark)/len(rows):.0%}), {len(bysrc)} sources, {(len(tiles)+PER-1)//PER} sheets")
for k, v in bysrc.most_common(8): print(f"  {k:22} {v:4d} dark of {allsrc[k]:4d}")
