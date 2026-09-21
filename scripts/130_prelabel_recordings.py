"""Model-assisted labelling on the DEVICE CAMERA's own footage (data/recording).

Why: every label we hold comes from other cameras. The v2 person model's
judgeability head, trained on data/sppg (2880x1620), says "hands visible" for
essentially everyone on this 704x576 camera, so it never abstains here.

    people      yolo11s-pose @0.40 over frames sampled across every clip,
                spread over both recorded days, capped per clip
    pre-labels  the v2 person model (presence + judgeability), the current candidate
    sheets      person | head zoom | hand-area zoom, 16 per sheet, ids printed

Output data/label_rec/
  people.csv    id, clip, frame time, box, pre-label + scores per item
  person/<id>.jpg
  sheets/sheet_NNN.jpg

    python scripts/130_prelabel_recordings.py [--n 2000]
"""
import argparse, collections, csv, json, random, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"deploy/apd_person")]
from apd.pose import build_rois
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--per-clip", type=int, default=4)
ap.add_argument("--frames-per-clip", type=int, default=4)
ap.add_argument("--min-h", type=int, default=70)
ap.add_argument("--out", default="data/label_rec")
a = ap.parse_args()
OUT = ROOT/a.out
for d in ("person", "sheets"): (OUT/d).mkdir(parents=True, exist_ok=True)
ITEMS = ["head_cap", "mask", "gloves"]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

j = json.loads((ROOT/"runs/person_ppe_v2/efficientnet_b0_320/summary.json").read_text())
net = torchvision.models.efficientnet_b0(weights=None)
net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 6)
net.load_state_dict(torch.load(j["weights"], map_location="cpu")); net = net.eval().cuda()
CUTS = j["cuts"]
pose = YOLO(str(ROOT/"yolo11s-pose.pt"))


def lb(im, s=320):
    h, w = im.shape[:2]; r = s/max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return ((cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)


clips = [p for p in sorted((ROOT/"data/recording").rglob("*.mp4")) if p.stat().st_size > 1e6]
rng = random.Random(20260918); rng.shuffle(clips)
print(f"{len(clips)} clips available")

rows, tiles = [], []
TH, ZW = 330, 165
col = lambda v: (0, 150, 0) if v == "present" else (0, 0, 220) if v == "absent" else (130, 130, 130)
short = {"present": "yes", "absent": "NO", "unknown": "?"}


def zoom(im, x1, y1, x2, y2, w, h):
    H, W = im.shape[:2]
    z = im[int(max(0, y1)):int(min(H, y2)), int(max(0, x1)):int(min(W, x2))]
    if z.size == 0: return np.full((h, w, 3), 235, np.uint8)
    s = min(w/z.shape[1], h/z.shape[0])
    z = cv2.resize(z, (max(1, int(z.shape[1]*s)), max(1, int(z.shape[0]*s))), interpolation=cv2.INTER_LANCZOS4)
    o = np.full((h, w, 3), 235, np.uint8); o[:z.shape[0], :z.shape[1]] = z
    return o


for clip in clips:
    if len(rows) >= a.n: break
    cap = cv2.VideoCapture(str(clip)); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(5) or 25
    if tot < 20: cap.release(); continue
    taken = 0
    for fi in np.linspace(tot*0.12, tot*0.92, a.frames_per_clip).astype(int):
        if taken >= a.per_clip or len(rows) >= a.n: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, im = cap.read()
        if not ok: continue
        H, W = im.shape[:2]
        r = pose.predict(im, imgsz=640, conf=0.40, verbose=False)[0]
        if not len(r.boxes): continue
        B = r.boxes.xyxy.cpu().numpy(); K = r.keypoints.data.cpu().numpy()
        ppl = [p for p in build_rois(B, K) if p.person_box[3]-p.person_box[1] >= a.min_h]
        if not ppl: continue
        crops, keep = [], []
        for pp in ppl:
            x1, y1, x2, y2 = pp.person_box; pw, ph = x2-x1, y2-y1
            c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
            if c.size and min(c.shape[:2]) >= 16: crops.append(c); keep.append(pp)
        if not crops: continue
        with torch.no_grad():
            P = torch.sigmoid(net(torch.from_numpy(np.stack([lb(c) for c in crops])).cuda())).cpu().numpy()
        order = np.argsort([-(p.person_box[3]-p.person_box[1]) for p in keep])
        for oi in order:
            if taken >= a.per_clip or len(rows) >= a.n: break
            pp, crop, pr = keep[oi], crops[oi], P[oi]
            n = len(rows); pid = f"{clip.stem}_f{int(fi)}_p{oi}"
            cv2.imwrite(str(OUT/"person"/f"{pid}.jpg"), crop)
            rec = {"n": n, "id": pid, "clip": clip.stem, "day": clip.parent.name if clip.parent.name != "recording" else "16",
                   "t_s": round(fi/max(fps, 1), 1), "box": json.dumps([int(v) for v in pp.person_box])}
            for k, it in enumerate(ITEMS):
                cut = CUTS[it]
                known = float(pr[3+k]) >= cut["known_above"]
                v = "unknown" if not known else ("absent" if float(pr[k]) < cut["present_below"] else "present")
                rec[it] = v; rec[it+"_p"] = round(float(pr[k]), 3); rec[it+"_known"] = round(float(pr[3+k]), 3)
            rows.append(rec); taken += 1

            s = TH/crop.shape[0]; pim = cv2.resize(crop, (max(1, int(crop.shape[1]*s)), TH), interpolation=cv2.INTER_LANCZOS4)
            PW = 190
            pim = pim[:, (pim.shape[1]-PW)//2:(pim.shape[1]-PW)//2+PW] if pim.shape[1] > PW else \
                cv2.copyMakeBorder(pim, 0, 0, 0, PW-pim.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            x1, y1, x2, y2 = pp.person_box; pw, ph = x2-x1, y2-y1
            side = max(0.32*ph, 0.8*pw)
            head = zoom(im, (x1+x2)/2-side/2, y1-0.08*ph, (x1+x2)/2+side/2, y1-0.08*ph+side, ZW, TH//2)
            hands = zoom(im, x1-0.15*pw, y1+0.30*ph, x2+0.15*pw, y1+0.85*ph, ZW, TH//2)
            t = cv2.copyMakeBorder(np.hstack([pim, np.vstack([head, hands])]), 30, 4, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, f"#{n}", (3, 21), 0, 0.6, (0, 0, 0), 2)
            for xx, it, lab in ((80, "head_cap", "cap"), (175, "mask", "mask"), (275, "gloves", "glv")):
                cv2.putText(t, f"{lab} {short[rec[it]]}", (xx, 21), 0, 0.55, col(rec[it]), 2)
            tiles.append(t)
    cap.release()
    if len(rows) % 200 < a.per_clip: print(f"  {len(rows)}/{a.n}", flush=True)

cols = ["n", "id", "clip", "day", "t_s", "box"] + [f(i) for i in ITEMS for f in (lambda s: s, lambda s: s+"_p", lambda s: s+"_known")]
with open(OUT/"people.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c: r[c] for c in cols})
PER, COLS = 16, 4
for s0 in range(0, len(tiles), PER):
    g = tiles[s0:s0+PER]; g += [np.full_like(tiles[0], 255)]*(PER-len(g))
    cv2.imwrite(str(OUT/"sheets"/f"sheet_{s0//PER:03d}.jpg"),
                np.vstack([np.hstack(g[i:i+COLS]) for i in range(0, PER, COLS)]), [cv2.IMWRITE_JPEG_QUALITY, 93])
c = collections.Counter((it, r[it]) for r in rows for it in ITEMS)
print(f"{len(rows)} people from {len({r['clip'] for r in rows})} clips, {(len(tiles)+PER-1)//PER} sheets")
print("pre-labels:", {f"{k[0]}:{k[1]}": v for k, v in sorted(c.items())})
