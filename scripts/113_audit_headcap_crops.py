"""Audit the head crops the deployed head_cap trunk learned from (data/crops_plus_sdk/head_cap).

One tile per unique head (augmented copies __h45/__h90/__h135/__g collapsed onto
their original), enlarged to 200 px, labelled with folder class, split and
source. Verdicts in verdicts.csv: ok | wrong_class | not_head | unclear.

  python scripts/113_audit_headcap_crops.py          -> artifacts/headcap_audit/sheet_NN.jpg + heads.csv
  python scripts/113_audit_headcap_crops.py --score
"""
import collections, csv, re, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/"data/crops_plus_sdk/head_cap"; OUT = ROOT/"artifacts/headcap_audit"
T, PER, COLS = 200, 30, 6


def source(b):
    return "sdk" if b.startswith("sdk") else "site" if re.match(r"\d", b) else "native"


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    heads = collections.defaultdict(list)
    for p in sorted(D.glob("*/*/*.jpg")):
        m = re.match(r"(.+?)__([a-z0-9]+)\.jpg$", p.name)
        b, suf = (m.group(1), m.group(2)) if m else (p.stem, "")
        heads[(p.parts[-3], b)].append((suf, p))
    rows, tiles = [], []
    for i, ((split, b), items) in enumerate(sorted(heads.items())):
        classes = sorted({p.parts[-2] for _, p in items})
        p = next((q for s, q in items if s in ("", "o")), items[0][1])
        lab = "CONFLICT" if len(classes) > 1 else ("cap" if classes[0] == "present" else "no_cap")
        col = {"cap": (0, 160, 0), "no_cap": (0, 0, 220), "CONFLICT": (200, 0, 200)}[lab]
        im = cv2.resize(cv2.imread(str(p)), (T, T), interpolation=cv2.INTER_LANCZOS4)
        t = cv2.copyMakeBorder(im, 24, 4, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, f"#{i} {lab}", (3, 17), 0, 0.55, col, 2)
        cv2.putText(t, f"{split[0]}/{source(b)[:4]}", (140, 17), 0, 0.4, (90, 90, 90), 1)
        tiles.append(t)
        rows.append([i, split, lab, source(b), b, len(items), "|".join(classes), str(p.relative_to(ROOT))])
    with open(OUT/"heads.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "split", "label", "source", "base", "copies", "folders", "crop"]); w.writerows(rows)
    for s in range(0, len(tiles), PER):
        g = tiles[s:s+PER]; g += [np.full_like(tiles[0], 255)]*(PER-len(g))
        cv2.imwrite(str(OUT/f"sheet_{s//PER:02d}.jpg"), np.vstack([np.hstack(g[k:k+COLS]) for k in range(0, PER, COLS)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(len(rows), "heads", collections.Counter((r[2], r[3]) for r in rows), "sheets", (len(tiles)+PER-1)//PER)


def score():
    h = {r["id"]: r for r in csv.DictReader(open(OUT/"heads.csv"))}
    v = {r["id"]: r["verdict"] for r in csv.DictReader(open(OUT/"verdicts.csv"))}
    for key in ("label", "source"):
        for val in sorted({r[key] for r in h.values()}) + ["all"]:
            ids = [i for i in h if val in ("all", h[i][key])]
            c = collections.Counter(v[i] for i in ids); n = len(ids); bad = c["wrong_class"] + c["not_head"]
            print(f"{key:6} {val:9} {n:5d}  " + "  ".join(f"{k} {c[k]}" for k in ("ok", "wrong_class", "not_head", "unclear"))
                  + f"  | wrong {bad/n:.0%}")


score() if "--score" in sys.argv else build()
