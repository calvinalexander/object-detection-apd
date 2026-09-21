"""Per-box audit sheets for the site-sourced gloves / no_gloves boxes in det7.

  python scripts/111_audit_site_gloves.py [--source site|voter3|sdk_apd_v3]   -> artifacts/det7_audit/gloves_<source>/
  python scripts/111_audit_site_gloves.py [--source ...] --score            -> joins verdicts.csv, prints error rates

--source picks which det7 crops to audit, by the frame's origin (the same
precedence scripts/92 used when a frame exists in several folders: SDK apd_v3
val first, then voter3, then the apd_v2/v3 dumps; consensus frames are "site").

One tile per box: the zoomed box (left) and the person crop with that box drawn
(right). Box ids are stable (sorted by file, then line), so verdicts.csv can be
written against the sheets: id,verdict with verdict in
  ok | wrong_class | not_hand | unclear      (not_hand = box not on the part the
  attribute is about: a hand for gloves, a face/mouth for mask, a head for head_cap)
  --attr picks the attribute (default gloves).
"""
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--source", default="site", choices=["site", "voter3", "sdk_apd_v3", "dump_apd_v2"])
ap.add_argument("--attr", default="gloves", choices=["gloves", "mask", "head_cap"])
ap.add_argument("--score", action="store_true")
ap.add_argument("--large", action="store_true", help="bigger tiles, 12 per sheet, with a context view (sheets go to <out>/large)")
A = ap.parse_args()
D = ROOT/"data/det7"; OUT = ROOT/f"artifacts/det7_audit/{A.attr}_{A.source}"
NATIVE = {"sdk_apd_v3": ROOT/"artifacts/sdk/sdk/customers/apd_v3/dataset", "voter3": ROOT/"data/voter3",
          "dump_apd_v2": ROOT/"data/dump/dataset/apd_v2/dataset", "dump_apd_v3": ROOT/"data/dump/dataset/apd_v3/dataset"}
ORIGIN = {}
for name, root in NATIVE.items():
    for sp in ("train", "val", "test"):
        if (root/"images"/sp).exists():
            for q in (root/"images"/sp).iterdir(): ORIGIN.setdefault(q.stem, name)
CLS = {"gloves": (1, 4), "head_cap": (2, 5), "mask": (3, 6)}
CONS = {json.loads(l)["id"] for l in open(ROOT/"artifacts/consensus.jsonl")}
SITES = {p.stem for sd in (ROOT/"data/sites").iterdir() if sd.is_dir() for p in sd.iterdir()}
T = 280 if "--large" in sys.argv else 200
POS, NEG = CLS[A.attr]
NAMES = {POS: A.attr, NEG: f"no_{A.attr}"}; COL = {POS: (0, 170, 0), NEG: (0, 0, 230)}


def rows():
    out = []
    for split in ("train", "val"):
        for lbl in sorted((D/f"labels/{split}").glob("*.txt")):
            f = lbl.stem.rsplit("_p", 1)[0]
            src = ORIGIN.get(f) or ("site" if f in CONS and f in SITES else None)
            if src != A.source: continue
            for ln, l in enumerate(lbl.read_text().splitlines()):
                if not l.strip() or int(l.split()[0]) not in NAMES: continue
                out.append(dict(split=split, stem=lbl.stem, line=ln, cls=int(l.split()[0]), yolo=[float(v) for v in l.split()[1:5]]))
    for i, r in enumerate(out): r["id"] = i
    return out


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    rs = rows(); tiles = []
    with open(OUT/"boxes.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "split", "image", "line", "class", "w_px", "h_px"])
        for r in rs:
            im = cv2.imread(str(D/f"images/{r['split']}/{r['stem']}.jpg")); H, W = im.shape[:2]
            x, y, bw, bh = r["yolo"]; x1, y1, x2, y2 = int((x-bw/2)*W), int((y-bh/2)*H), int((x+bw/2)*W), int((y+bh/2)*H)
            w.writerow([r["id"], r["split"], r["stem"], r["line"], NAMES[r["cls"]], x2-x1, y2-y1])
            p = max(6, (x2-x1)//3)
            z = im[max(0, y1-p):y2+p, max(0, x1-p):x2+p].copy()
            cv2.rectangle(z, (x1-max(0, x1-p), y1-max(0, y1-p)), (x2-max(0, x1-p), y2-max(0, y1-p)), COL[r["cls"]], 1)
            z = cv2.resize(z, (T, T), interpolation=cv2.INTER_LANCZOS4 if A.large else cv2.INTER_CUBIC)
            c = im.copy(); cv2.rectangle(c, (x1, y1), (x2, y2), COL[r["cls"]], 2)
            cw = 180 if A.large else 130
            c = cv2.resize(c, (int(T*W/H), T)); c = c[:, :cw] if c.shape[1] > cw else cv2.copyMakeBorder(c, 0, 0, 0, cw-c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            if A.large:
                # wider context: 3x the box around its centre, so a face is judged with its head and shoulders
                cx, cy, R = (x1+x2)//2, (y1+y2)//2, int(max(x2-x1, y2-y1)*1.6)
                ctx = im[max(0, cy-R):cy+R, max(0, cx-R):cx+R].copy()
                cv2.rectangle(ctx, (x1-max(0, cx-R), y1-max(0, cy-R)), (x2-max(0, cx-R), y2-max(0, cy-R)), COL[r["cls"]], 1)
                ctx = cv2.resize(ctx, (T, T), interpolation=cv2.INTER_LANCZOS4)
                t = np.hstack([z, ctx, c])
            else:
                t = np.hstack([z, c])
            t = cv2.copyMakeBorder(t, 24, 4, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, f"#{r['id']} {NAMES[r['cls']]} {x2-x1}x{y2-y1}", (4, 18), 0, 0.55, COL[r["cls"]], 2)
            tiles.append(t)
    per, cols = (12, 3) if A.large else (24, 6)
    sd = OUT/"large" if A.large else OUT; sd.mkdir(exist_ok=True)
    for s in range(0, len(tiles), per):
        g = tiles[s:s+per]; g += [np.full_like(tiles[0], 255)]*(per-len(g))
        cv2.imwrite(str(sd/f"sheet_{s//per:02d}.jpg"), np.vstack([np.hstack(g[i:i+cols]) for i in range(0, per, cols)]), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(len(rs), "boxes", {n: sum(r["cls"] == c for r in rs) for c, n in NAMES.items()}, "sheets:", (len(tiles)+per-1)//per)


def score():
    box = {r["id"]: r for r in csv.DictReader(open(OUT/"boxes.csv"))}
    ver = {r["id"]: r["verdict"] for r in csv.DictReader(open(OUT/"verdicts.csv"))}
    for cls in (*NAMES.values(), "all"):
        ids = [i for i, b in box.items() if cls in ("all", b["class"])]
        c = {v: sum(ver[i] == v for i in ids) for v in ("ok", "wrong_class", "not_hand", "unclear")}
        n = len(ids); bad = c["wrong_class"] + c["not_hand"]
        print(f"{cls:10} {n:4d} boxes  " + "  ".join(f"{k} {v} ({v/n:.0%})" for k, v in c.items()) + f"  | wrong {bad/n:.0%}")


score() if A.score else build()
