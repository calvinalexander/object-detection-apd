"""Audit sheets for data/label_sppg (made by scripts/119), redrawn larger for reviewers.

Tile: person (+18%) | head zoom (top 32% of the person box, square) over hand zoom
(person box rows 30-85%, where hands are). Pre-labels printed on top. 16 tiles/sheet.

    python scripts/120_label_sheets_sppg.py
"""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/(sys.argv[1] if len(sys.argv) > 1 else "data/label_sppg"); S = D/"sheets"
S.mkdir(parents=True, exist_ok=True)
REC = (D.name == "label_recording")        # recordings: frames come from a clip, not data/sppg
rows = list(csv.DictReader(open(D/"people.csv")))
for f in S.glob("*.jpg"): f.unlink()
TH, ZW, PW = 330, 165, 190
col = lambda v: (0, 150, 0) if v == "present" else (0, 0, 220) if v == "absent" else (130, 130, 130)
short = {"present": "yes", "absent": "NO", "unknown": "?"}
tiles, cache = [], {}


def zoom(im, x1, y1, x2, y2, w, h):
    H, W = im.shape[:2]
    z = im[int(max(0, y1)):int(min(H, y2)), int(max(0, x1)):int(min(W, x2))]
    if z.size == 0: return np.full((h, w, 3), 235, np.uint8)
    s = min(w/z.shape[1], h/z.shape[0]); z = cv2.resize(z, (max(1, int(z.shape[1]*s)), max(1, int(z.shape[0]*s))), interpolation=cv2.INTER_LANCZOS4)
    out = np.full((h, w, 3), 235, np.uint8); out[:z.shape[0], :z.shape[1]] = z
    return out


for r in rows:
    if REC:
        kfn = (r["clip"], r["sec"])
        if kfn not in cache:
            v = next(ROOT.glob(f"data/recording/**/{r['clip']}.mp4"))
            cap = cv2.VideoCapture(str(v)); cap.set(cv2.CAP_PROP_POS_MSEC, float(r["sec"])*1000)
            ok, fr = cap.read(); cache = {kfn: fr if ok else None}
    else:
        kfn = r["frame"]
        if kfn not in cache: cache = {kfn: cv2.imread(str(ROOT/"data/sppg"/kfn))}
    im = cache[kfn]
    if im is None: continue
    x1, y1, x2, y2 = json.loads(r["box"]); pw, ph = x2-x1, y2-y1
    person = cv2.imread(str(D/"person"/f"{r['id']}.jpg"))
    if person is None: continue
    s = TH/person.shape[0]; p = cv2.resize(person, (max(1, int(person.shape[1]*s)), TH), interpolation=cv2.INTER_LANCZOS4)
    p = p[:, (p.shape[1]-PW)//2:(p.shape[1]-PW)//2+PW] if p.shape[1] > PW else cv2.copyMakeBorder(p, 0, 0, 0, PW-p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    side = max(0.32*ph, 0.8*pw)
    head = zoom(im, (x1+x2)/2-side/2, y1-0.08*ph, (x1+x2)/2+side/2, y1-0.08*ph+side, ZW, TH//2)
    hands = zoom(im, x1-0.15*pw, y1+0.30*ph, x2+0.15*pw, y1+0.85*ph, ZW, TH//2)
    t = cv2.copyMakeBorder(np.hstack([p, np.vstack([head, hands])]), 30, 4, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(t, f"#{r['n']}", (3, 21), 0, 0.6, (0, 0, 0), 2)
    for x, it, lab in ((80, "head_cap", "cap"), (175, "mask", "mask"), (275, "gloves", "glv")):
        cv2.putText(t, f"{lab} {short[r[it]]}", (x, 21), 0, 0.55, col(r[it]), 2)
    tiles.append(t)
PER, COLS = 16, 4
for s0 in range(0, len(tiles), PER):
    g = tiles[s0:s0+PER]; g += [np.full_like(tiles[0], 255)]*(PER-len(g))
    cv2.imwrite(str(S/f"sheet_{s0//PER:03d}.jpg"), np.vstack([np.hstack(g[j:j+COLS]) for j in range(0, PER, COLS)]),
                [cv2.IMWRITE_JPEG_QUALITY, 93])
print(len(tiles), "tiles,", (len(tiles)+PER-1)//PER, "sheets")
