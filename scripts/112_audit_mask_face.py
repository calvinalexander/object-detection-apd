"""Audit the crops the deployed mask trunk learned from (data/mask_face/g0.8).

One tile per labelled face: the 128 px training crop (what the model saw), the
annotator's face box in its frame, and the frame. Only faces whose frame is in
data/voter3 are audited (production's SDK val frames and COCO are left out on
purpose). Verdicts in verdicts.csv: ok | wrong_class | wrong_face | unclear
(wrong_face = the crop is not the labelled face: pose matched another person or
the head region missed).

  python scripts/112_audit_mask_face.py [--all]           -> artifacts/mask_face_audit/{voter3|all}/sheet_NN.jpg, faces.csv
  python scripts/112_audit_mask_face.py [--all] --score

--all audits every face in the set, whatever its source.
"""
import csv, re, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALL = "--all" in sys.argv
MF = ROOT/"data/mask_face/g0.8"; OUT = ROOT/"artifacts/mask_face_audit"/("all" if ALL else "voter3")
N = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
SOURCES = [ROOT/"artifacts/sdk/sdk/customers/apd_v3/dataset", ROOT/"data/dump/dataset/apd_v2/dataset",
           ROOT/"data/dump/dataset/apd_v3/dataset", ROOT/"data/voter3"]
V3 = {p.stem for p in (ROOT/"data/voter3").rglob("*.txt") if "labels" in p.parts}
T = 240


def frames():
    """stem -> (image, faces) exactly as scripts/101 resolved them (first source wins)."""
    out = {}
    for src in SOURCES:
        for lp in src.rglob("*.txt"):
            if "labels" not in lp.parts or lp.stem in out: continue
            faces = [(N[int(f[0])] == "mask", *map(float, f[1:5])) for f in (l.split() for l in lp.read_text().splitlines())
                     if len(f) >= 5 and N[int(f[0])] in ("mask", "no_mask")]
            img = Path(str(lp.parent).replace("labels", "images"))/(lp.stem + ".jpg")
            if faces and img.exists(): out[lp.stem] = (img, faces)
    return out


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    fr = frames(); by60 = {s[:60]: s for s in fr}
    crops = {}
    for p in MF.rglob("*.jpg"):
        m = re.match(r"(.+)_(\d+)_([nx])(\d)\.jpg$", p.name)
        s = by60.get(m.group(1))
        if s is None or (not ALL and s not in V3) or m.group(4) != "0": continue
        key = (s, int(m.group(2)))
        if key not in crops or m.group(3) == "n":             # prefer the device's own pose model
            crops[key] = (p, p.parts[-3], p.parts[-2], m.group(3))
    rows, tiles = [], []
    for i, ((s, fi), (p, split, cls, pm)) in enumerate(sorted(crops.items())):
        img, faces = fr[s]; im = cv2.imread(str(img)); H, W = im.shape[:2]
        present, cx, cy, bw, bh = faces[fi]
        x1, y1, x2, y2 = int((cx-bw/2)*W), int((cy-bh/2)*H), int((cx+bw/2)*W), int((cy+bh/2)*H)
        col = (0, 170, 0) if present else (0, 0, 230)
        c = cv2.resize(cv2.imread(str(p)), (T, T), interpolation=cv2.INTER_LANCZOS4)
        R = int(max(x2-x1, y2-y1)*1.8); mx, my = (x1+x2)//2, (y1+y2)//2
        z = im[max(0, my-R):my+R, max(0, mx-R):mx+R].copy()
        cv2.rectangle(z, (x1-max(0, mx-R), y1-max(0, my-R)), (x2-max(0, mx-R), y2-max(0, my-R)), col, 1)
        z = cv2.resize(z, (T, T), interpolation=cv2.INTER_LANCZOS4)
        f = im.copy(); cv2.rectangle(f, (x1, y1), (x2, y2), col, 3)
        f = cv2.resize(f, (int(T*W/H), T))[:, :180]
        f = cv2.copyMakeBorder(f, 0, 0, 0, 180-f.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        t = cv2.copyMakeBorder(np.hstack([c, z, f]), 26, 4, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, f"#{i} {'mask' if present else 'no_mask'} {split} {x2-x1}x{y2-y1}", (4, 19), 0, 0.6, col, 2)
        tiles.append(t)
        rows.append([i, split, "mask" if present else "no_mask", s, fi, pm, x2-x1, y2-y1, str(p.relative_to(ROOT))])
    with open(OUT/"faces.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "split", "label", "frame", "face", "pose", "w_px", "h_px", "crop"]); w.writerows(rows)
    for s in range(0, len(tiles), 12):
        g = tiles[s:s+12]; g += [np.full_like(tiles[0], 255)]*(12-len(g))
        cv2.imwrite(str(OUT/f"sheet_{s//12:02d}.jpg"), np.vstack([np.hstack(g[k:k+3]) for k in range(0, 12, 3)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    import collections
    print(len(rows), "faces", collections.Counter((r[1], r[2]) for r in rows), "sheets", (len(tiles)+11)//12)


def score():
    import collections
    f = {r["id"]: r for r in csv.DictReader(open(OUT/"faces.csv"))}
    v = {r["id"]: r["verdict"] for r in csv.DictReader(open(OUT/"verdicts.csv"))}
    for lab in ("mask", "no_mask", "all"):
        ids = [i for i in f if lab in ("all", f[i]["label"])]
        c = collections.Counter(v[i] for i in ids); n = len(ids); bad = c["wrong_class"] + c["wrong_face"]
        print(f"{lab:8} {n:4d} faces  " + "  ".join(f"{k} {c[k]}" for k in ("ok", "wrong_class", "wrong_face", "unclear"))
              + f"  | wrong {bad/n:.0%}")


score() if "--score" in sys.argv else build()
