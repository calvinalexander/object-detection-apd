"""Sample det7 training crops for the two gloves classes, with a zoom on each hand box.

  python scripts/110_viz_det7_gloves.py              -> artifacts/det7_viz/07_gloves/{gloves,no_gloves}_NN.jpg
  python scripts/110_viz_det7_gloves.py --source site  -> artifacts/det7_viz/07_gloves_site/...

--source site keeps crops of production snapshots (data/sites, labels from
artifacts/consensus.jsonl), the same split scripts/93 calls "site";
--source native keeps the frames that arrived already annotated.
"""
import argparse, json, random
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--source", choices=["all", "site", "native"], default="all")
ap.add_argument("--split", default="train"); a = ap.parse_args()
D = ROOT/"data/det7"; OUT = ROOT/("artifacts/det7_viz/07_gloves" + ("" if a.source == "all" else f"_{a.source}"))
OUT.mkdir(parents=True, exist_ok=True)
CONS = {json.loads(l)["id"] for l in open(ROOT/"artifacts/consensus.jsonl")}
SITES = {p.stem for sd in (ROOT/"data/sites").iterdir() if sd.is_dir() for p in sd.iterdir()}


def keep(stem):
    site = stem.rsplit("_p", 1)[0] in CONS and stem.rsplit("_p", 1)[0] in SITES
    return a.source == "all" or (a.source == "site") == site
NAMES = {1: "gloves", 4: "no_gloves"}; COL = {1: (0, 200, 0), 4: (0, 0, 255)}
H = 260; ZOOM = 120


def boxes(lbl, W, Hh):
    out = []
    for l in lbl.read_text().splitlines():
        c, x, y, w, h = l.split()[:5]; c = int(c)
        if c in NAMES:
            x, y, w, h = float(x)*W, float(y)*Hh, float(w)*W, float(h)*Hh
            out.append((c, int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)))
    return out


def tile(img, bx, want):
    s = H/img.shape[0]
    full = cv2.resize(img, (int(img.shape[1]*s), H))
    for c, x1, y1, x2, y2 in bx:
        cv2.rectangle(full, (int(x1*s), int(y1*s)), (int(x2*s), int(y2*s)), COL[c], 2)
    full = full[:, :180] if full.shape[1] > 180 else cv2.copyMakeBorder(full, 0, 0, 0, 180-full.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    zs = []
    for c, x1, y1, x2, y2 in [b for b in bx if b[0] == want][:2]:
        p = max(4, (x2-x1)//4)
        z = img[max(0, y1-p):y2+p, max(0, x1-p):x2+p]
        z = cv2.resize(z, (ZOOM, ZOOM), interpolation=cv2.INTER_CUBIC)
        cv2.putText(z, f"{x2-x1}x{y2-y1}px", (3, ZOOM-5), 0, 0.4, (255, 255, 255), 2)
        cv2.putText(z, f"{x2-x1}x{y2-y1}px", (3, ZOOM-5), 0, 0.4, (0, 0, 0), 1)
        zs.append(cv2.copyMakeBorder(z, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=COL[want]))
    while len(zs) < 2: zs.append(np.full((ZOOM+4, ZOOM+4, 3), 255, np.uint8))
    right = np.vstack(zs); right = cv2.copyMakeBorder(right, 0, H-right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.copyMakeBorder(np.hstack([full, right]), 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))


rng = random.Random(0)
for split in (a.split,):
    items = {1: [], 4: []}
    for lbl in sorted((D/f"labels/{split}").glob("*.txt")):
        if not keep(lbl.stem): continue
        cls = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
        for c in items:
            if c in cls and not ({1, 4} - {c}) & cls:      # only one of the two classes on the person
                items[c].append(lbl)
    for c, lst in items.items():
        rng.shuffle(lst)
        for page in range(2):
            if page*20 >= len(lst): break
            tiles = []
            for lbl in lst[page*20:(page+1)*20]:
                img = cv2.imread(str(D/f"images/{split}/{lbl.stem}.jpg"))
                tiles.append(tile(img, boxes(lbl, img.shape[1], img.shape[0]), c))
            tiles += [np.full_like(tiles[0], 255)]*(-len(tiles) % 5)
            grid = np.vstack([np.hstack(tiles[i:i+5]) for i in range(0, len(tiles), 5)])
            head = np.full((36, grid.shape[1], 3), 255, np.uint8)
            cv2.putText(head, f"det7 {split} [{a.source}]: {NAMES[c]}  ({len(lst)} crops with only this class)  green=gloves red=no_gloves",
                        (8, 25), 0, 0.7, (0, 0, 0), 2)
            cv2.imwrite(str(OUT/f"{NAMES[c]}_{page+1:02d}.jpg"), np.vstack([head, grid]))
    print({NAMES[c]: len(v) for c, v in items.items()})
