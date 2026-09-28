"""Audit sheets for everything v3 trains on: one person per tile, 6 per sheet.

Tile: person crop (as the model sees it, letterboxed 320) | head zoom | hand zoom,
with the labels the training set currently carries printed on top.

Sources, in the order they are audited:
  det7_export   det7 crops from the dashboard exports  (never audited)
  det7_site     det7 crops from the site clips         (gloves/mask partly audited)
  det7_cam1     det7 crops from the cam1 clips         (never audited)
  sppg          data/label_sppg                        (reviewed once, from zooms)
  coco          data/clean/person_coco                 (human labels, never re-checked)

    python scripts/144_v3_audit_sheets.py <source> [--start 0] [--n 60]
Writes artifacts/v3_audit/<source>/sheet_NNN.jpg and index.csv
"""
import argparse, csv, glob, json, re
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/v3_audit"
ITEMS = ["head_cap", "mask", "gloves"]
CLS = {"head_cap": (2, 5), "mask": (3, 6), "gloves": (1, 4)}
ap = argparse.ArgumentParser()
ap.add_argument("source", choices=["det7_export", "det7_site", "det7_cam1", "sppg", "coco"])
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--per", type=int, default=12)
a = ap.parse_args()


def det7_rows(kind):
    out = []
    for sp in ("train", "val"):
        for lp in sorted((ROOT/f"data/clean/det7/labels/{sp}").glob("*.txt")):
            n = lp.stem
            k = ("export" if re.match(r"^\d{9,}_p\d+$", n) else
                 "site" if re.match(r"^[0-9a-f]{8}-frame_", n) else "cam1")
            if k != kind: continue
            cls = {int(l.split()[0]) for l in lp.read_text().splitlines() if l.strip()}
            lab = {}
            for it, (pos, neg) in CLS.items():
                lab[it] = "present" if pos in cls else "absent" if neg in cls else "-"
            out.append({"id": n, "img": str(ROOT/f"data/clean/det7/images/{sp}/{n}.jpg"), **lab})
    return out


def csv_rows(path, img_key):
    out = []
    for r in csv.DictReader(open(path)):
        out.append({"id": r.get("id") or r.get("n"), "img": str(ROOT/r[img_key]),
                    **{i: r[i] for i in ITEMS}})
    return out


rows = (det7_rows(a.source.split("_")[1]) if a.source.startswith("det7") else
        csv_rows(ROOT/"data/label_sppg/labels.csv", "person") if a.source == "sppg" else
        csv_rows(ROOT/"data/clean/person_coco/labels.csv", "person"))
rows = rows[a.start:a.start+a.n]
d = OUT/a.source; d.mkdir(parents=True, exist_ok=True)
SH = {"present": "YES", "absent": "NO", "unclear": "?", "-": "-", "not_person": "X"}
COL = {"present": (0, 140, 0), "absent": (0, 0, 200), "unclear": (130, 130, 130),
       "-": (150, 150, 150), "not_person": (0, 0, 0)}
PW, ZW, TH = 210, 200, 400


def zoom(im, y0, y1, w, h):
    c = im[int(max(0, y0*im.shape[0])):int(min(im.shape[0], y1*im.shape[0]))]
    if c.size == 0: return np.full((h, w, 3), 235, np.uint8)
    s = min(w/c.shape[1], h/c.shape[0])
    c = cv2.resize(c, (max(1, int(c.shape[1]*s)), max(1, int(c.shape[0]*s))), interpolation=cv2.INTER_CUBIC)
    o = np.full((h, w, 3), 235, np.uint8); o[:c.shape[0], :c.shape[1]] = c
    return o


tiles, index = [], []
for r in rows:
    im = cv2.imread(r["img"])
    if im is None: continue
    s = TH/im.shape[0]
    p = cv2.resize(im, (max(1, int(im.shape[1]*s)), TH), interpolation=cv2.INTER_CUBIC)
    p = (p[:, (p.shape[1]-PW)//2:(p.shape[1]-PW)//2+PW] if p.shape[1] > PW else
         cv2.copyMakeBorder(p, 0, 0, 0, PW-p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    t = np.hstack([p, np.vstack([zoom(im, 0, 0.34, ZW, TH//2), zoom(im, 0.30, 0.90, ZW, TH//2)])])
    t = cv2.copyMakeBorder(t, 34, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(t, f"#{len(index)+a.start}", (5, 14), 0, 0.45, (0, 0, 0), 1)
    cv2.putText(t, f"{im.shape[1]}x{im.shape[0]}", (5, 29), 0, 0.42, (110, 110, 110), 1)
    x = 70
    for it in ITEMS:
        txt = f"{it[:3]} {SH.get(r[it], r[it])}"
        cv2.putText(t, txt, (x, 22), 0, 0.5, COL.get(r[it], (0, 0, 0)), 2); x += 12*len(txt)
    tiles.append(t)
    index.append({"n": len(index)+a.start, "source": a.source, "id": r["id"],
                  **{i: r[i] for i in ITEMS}, "img": r["img"]})

for s0 in range(0, len(tiles), a.per):
    g = tiles[s0:s0+a.per]
    while len(g) < a.per: g.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(d/f"sheet_{(a.start+s0)//a.per:04d}.jpg"),
                np.vstack([np.hstack(g[i:i+3]) for i in range(0, a.per, 3)]), [cv2.IMWRITE_JPEG_QUALITY, 95])
mode = "a" if (d/"index.csv").exists() and a.start else "w"
with open(d/"index.csv", mode, newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(index[0]))
    if mode == "w": w.writeheader()
    w.writerows(index)
print(f"{a.source}: {len(index)} people -> {(len(tiles)+a.per-1)//a.per} sheets in {d}")
