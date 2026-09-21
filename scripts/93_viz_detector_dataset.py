"""Render the detector dataset into folders you can page through.

One folder per question a reviewer actually asks:

    01_train / 02_val        what the model will see, boxes drawn
    03_by_class/<class>      is this class boxed consistently?
    04_overlay_cleaned       did the dashboard caption really go?
    06_by_source/*          native annotator boxes against production-snapshot labels
"""
import argparse, collections, json, random, shutil
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
GREEN, RED, WHITE = (60, 200, 60), (60, 60, 235), (230, 230, 230)
COLOR = {0: WHITE, 1: GREEN, 2: GREEN, 3: GREEN, 4: RED, 5: RED, 6: RED}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/det7")
ap.add_argument("--out", default="artifacts/det7_viz")
ap.add_argument("--per-folder", type=int, default=150)
ap.add_argument("--width", type=int, default=1100)
ap.add_argument("--grid", type=int, default=1, help="person crops are small: show them 4x3 per sheet")
a = ap.parse_args()
DATA, OUT = ROOT/a.data, ROOT/a.out
if OUT.exists(): shutil.rmtree(OUT)
random.seed(20260916)


def read(split, stem):
    lp = DATA/"labels"/split/f"{stem}.txt"
    if not lp.exists(): return []
    out = []
    for line in lp.read_text().splitlines():
        f = line.split()
        if len(f) >= 5: out.append((int(f[0]), *[float(v) for v in f[1:5]]))
    return out


def draw(im, boxes, highlight=None):
    H, W = im.shape[:2]
    scale = max(0.4, min(0.75, W/1600))
    th = 2 if W > 900 else 1
    for c, cx, cy, bw, bh in sorted(boxes, key=lambda b: -b[3]*b[4]):
        x1, y1 = int((cx-bw/2)*W), int((cy-bh/2)*H)
        x2, y2 = int((cx+bw/2)*W), int((cy+bh/2)*H)
        dim = highlight is not None and c != highlight
        col = tuple(int(v*0.35) for v in COLOR[c]) if dim else COLOR[c]
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 1 if (c == 0 or dim) else th)
        if dim: continue
        txt = NAMES[c]
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = max(tht+2, y1-3)
        cv2.rectangle(im, (x1, ty-tht-2), (x1+tw+4, ty+2), (0, 0, 0), -1)
        cv2.putText(im, txt, (x1+2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
    return im


def fit(im, w=None):
    w = w or a.width
    h = int(im.shape[0]*w/im.shape[1])
    return cv2.resize(im, (w, h))


def banner(im, text):
    bar = np.full((28, im.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


# ---------------------------------------------------------------- collect
items = {}
for split in ("train", "val"):
    for p in sorted((DATA/"images"/split).iterdir()):
        items[p] = (split, read(split, p.stem))
print(f"{len(items)} images in {a.data}")

consensus_ids = {json.loads(l)["id"] for l in open(ROOT/"artifacts/consensus.jsonl")}
PERSON_MODE = any("_p" in p.stem.rsplit("_", 1)[-1][:2] or p.stem.rsplit("_p", 1)[-1].isdigit()
                  for p in list(items)[:20]) if False else all(
    "_p" in p.stem and p.stem.rsplit("_p", 1)[-1].isdigit() for p in list(items)[:20])
print("person-crop dataset" if PERSON_MODE else "full-frame dataset")


def sheet(paths, cols=4, rows=3, cell=(300, 400)):
    """A contact sheet of person crops, each with its own boxes drawn."""
    tiles = []
    for p in paths[:cols*rows]:
        im = cv2.imread(str(p))
        if im is None: continue
        split, boxes = items[p]
        im = draw(im.copy(), boxes)
        h, w = im.shape[:2]
        sc = min(cell[0]/w, cell[1]/h)
        im = cv2.resize(im, (max(1, int(w*sc)), max(1, int(h*sc))))
        pad = np.full((cell[1], cell[0], 3), 18, np.uint8)
        y0, x0 = (cell[1]-im.shape[0])//2, (cell[0]-im.shape[1])//2
        pad[y0:y0+im.shape[0], x0:x0+im.shape[1]] = im
        n = collections.Counter(NAMES[c] for c, *_ in boxes if c != 0)
        cv2.putText(pad, p.stem[-22:], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(pad, " ".join(f"{k}" for k in n) or "person only", (4, cell[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 220, 170), 1, cv2.LINE_AA)
        tiles.append(pad)
    if not tiles: return None
    while len(tiles) % cols: tiles.append(np.full((cell[1], cell[0], 3), 18, np.uint8))
    return np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)])


def emit(folder, pool, label):
    d = OUT/folder; d.mkdir(parents=True, exist_ok=True)
    random.shuffle(pool)
    if PERSON_MODE and a.grid:
        per = 12
        for k in range(0, min(len(pool), a.per_folder), per):
            sh = sheet(pool[k:k+per])
            if sh is None: continue
            cv2.imwrite(str(d/f"{label}_{k//per:03d}.jpg"), banner(sh, f"{label}  sheet {k//per+1}"))
    else:
        for p in pool[:a.per_folder]:
            im = cv2.imread(str(p))
            if im is None: continue
            split, boxes = items[p]
            n = collections.Counter(NAMES[c] for c, *_ in boxes)
            cv2.imwrite(str(d/p.name), banner(fit(draw(im, boxes)),
                        f"{label}  {split}  {p.name}   " + "  ".join(f"{k}x{v}" for k, v in n.most_common())))
    return len(list(d.iterdir()))
sites = {p.stem: p for sd in (ROOT/"data/sites").iterdir() if sd.is_dir() for p in sd.iterdir()}

# ---------------------------------------------------------------- 01 / 02 splits
for split, folder in (("train", "01_train"), ("val", "02_val")):
    pool = [p for p, (sp, b) in items.items() if sp == split and (PERSON_MODE or len(b) >= 3)]
    print(f"  {folder}: {len(pool)} images, {emit(folder, pool, split)} written")

# ---------------------------------------------------------------- 03 per class
for cid, cname in enumerate(NAMES):
    pool = [p for p, (sp, b) in items.items() if any(c == cid for c, *_ in b)]
    n = emit(f"03_by_class/{cid}_{cname}", pool, cname)
    print(f"  03_by_class/{cid}_{cname}: {len(pool)} contain it, {n} written")

# ---------------------------------------------------------------- 04 overlay before/after
if not PERSON_MODE:
    d = OUT/"04_overlay_cleaned"; d.mkdir(parents=True, exist_ok=True)
    pool = [p for p in items if p.stem in consensus_ids and p.stem in sites]
    random.shuffle(pool)
    for p in pool[:a.per_folder]:
        ds, src = cv2.imread(str(p)), cv2.imread(str(sites[p.stem]))
        if ds is None or src is None: continue
        wd = a.width//2
        cv2.imwrite(str(d/p.name), np.hstack([
            banner(fit(src, wd), "source: dashboard caption drawn on"),
            banner(fit(draw(ds, items[p][1]), wd), "dataset: caption removed, boxes")]))
    print(f"  04_overlay_cleaned: {len(list(d.iterdir()))}")

# ---------------------------------------------------------------- 06 by source
orig = {}
for line in open(ROOT/"artifacts/consensus.jsonl"):
    c = json.loads(line); n = 0
    for pp in c["persons"]:
        n += 1
        for at, v in pp.get("attrs", {}).items():
            if v.get("decision") in ("present", "absent") and v.get("box"): n += 1
    orig[c["id"]] = n

def frame_stem(p):
    return p.stem.rsplit("_p", 1)[0] if PERSON_MODE else p.stem

buckets = {"native": [], "site": []}
for p, (split, boxes) in items.items():
    buckets["site" if frame_stem(p) in consensus_ids else "native"].append(p)
if not PERSON_MODE:
    site = buckets.pop("site")
    buckets["site_intact"] = [p for p in site if len(items[p][1]) >= orig.get(p.stem, 0)]
    buckets["site_partial"] = [p for p in site if len(items[p][1]) < orig.get(p.stem, 0)]
for bname, pool in buckets.items():
    n = emit(f"06_by_source/{bname}", pool, bname)
    print(f"  06_by_source/{bname}: {len(pool)} images, {n} written")

# ---------------------------------------------------------------- index
cnt = {s: collections.Counter() for s in ("train", "val")}
for p, (s, b) in items.items():
    for c, *_ in b: cnt[s][NAMES[c]] += 1
lines = ["# det7 dataset visualisation", "",
         "Green is a satisfied attribute, red is a violation, white is the person box.", "",
         "| class | train | val |", "|---|---|---|"]
for n in NAMES: lines.append(f"| {n} | {cnt['train'][n]} | {cnt['val'][n]} |")
lines += ["", f"Images: {sum(1 for _,(s,_) in items.items() if s=='train')} train, "
              f"{sum(1 for _,(s,_) in items.items() if s=='val')} val"
              + ("  (one per person, cropped from the frame)" if PERSON_MODE else ""), "",
          "| folder | shows |", "|---|---|",
          "| 01_train / 02_val | what the model sees, all boxes drawn |",
          "| 03_by_class | one folder per class, that class highlighted and the rest dimmed |",
          "| 04_overlay_cleaned | source snapshot beside the dataset copy, caption removed |",
          "| 06_by_source/native | crops from frames that arrived already annotated |",
          "| 06_by_source/site | crops from production snapshots, consensus labels |"]
(OUT/"README.md").write_text("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
