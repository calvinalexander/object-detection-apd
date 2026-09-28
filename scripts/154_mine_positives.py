"""Mine production's own `ok` verdicts as weak POSITIVE labels.

The training set holds only 642 examples of a mask correctly worn and 773 of
gloves worn -- which is why v5 calls a quarter of properly masked workers at the
23 Sep site unmasked. Production's `apd_detail` table records an `ok` verdict per
item per event, on both cameras, and those are exactly the class we lack.

They are WEAK labels: production is wrong often enough that they need review
before training (its head_cap alerts at the 23 Sep site look mostly wrong). So
this writes crops plus review sheets, and records our own model's score beside
each one, so the disagreements can be looked at first.

Dedup: at most --per-track crops per (camera, track_id), spaced --gap seconds.

    python scripts/154_mine_positives.py [--per-track 2] [--min-h 60]
Writes data/mined_positives/{<item>/<id>.jpg, candidates.csv, sheets/}
"""
import argparse, csv, collections, datetime as dt, json, sqlite3
from pathlib import Path
import cv2, numpy as np, torch, torchvision

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--per-track", type=int, default=2)
ap.add_argument("--gap", type=float, default=60)
ap.add_argument("--min-h", type=int, default=60)
ap.add_argument("--run", default="runs/person_ppe_v5/efficientnet_b0_320")
ap.add_argument("--out", default="data/mined_positives")
a = ap.parse_args()
ITEMS = ["head_cap", "mask", "gloves"]
NAME = {"headcap": "head_cap", "mask": "mask", "gloves": "gloves"}
OUT = ROOT/a.out
for i in ITEMS: (OUT/i).mkdir(parents=True, exist_ok=True)
(OUT/"sheets").mkdir(exist_ok=True)
SITES = [("dev16", ROOT/"artifacts/replay16_person/prod_device/apd_storage/apd_events.db",
          ROOT/"artifacts/replay16_person/prod_device", True),
         ("site23", ROOT/"data/snapshot/23/db", ROOT/"data/snapshot/23", False)]

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
j = json.loads((ROOT/a.run/"summary.json").read_text())
net = getattr(torchvision.models, j["arch"])(weights=None)
net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 6)
net.load_state_dict(torch.load(ROOT/a.run/"best.pt", map_location="cpu", weights_only=False))
net = net.eval().cuda()


def prep(c, s=320):
    h, w = c.shape[:2]; r = s/max(h, w)
    im = cv2.resize(c, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return ((cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)


rows, seen = [], collections.defaultdict(list)
for site, dbp, base, rel in SITES:
    db = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
    ok = collections.defaultdict(set)
    for e, p, s in db.execute("select event_id,ppe_type,status from apd_detail"):
        if s == "ok": ok[e].add(NAME.get(p, p))
    for eid, ts, tid, bbox, ip in db.execute("select event_id,timestamp,track_id,person_bbox,image_path from events"):
        items = ok.get(eid)
        if not items: continue
        t = dt.datetime.fromisoformat(ts)
        key = (site, tid)
        if len(seen[key]) >= a.per_track: continue
        if any(abs((t-p).total_seconds()) < a.gap for p in seen[key]): continue
        f = (base/ip) if rel else (base/f"{eid}.jpg")
        im = cv2.imread(str(f))
        if im is None: continue
        H, W = im.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in json.loads(bbox)]
        if y2-y1 < a.min_h: continue
        pw, ph = x2-x1, y2-y1
        c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
        if c.size == 0 or min(c.shape[:2]) < 16: continue
        seen[key].append(t)
        rows.append({"id": f"{site}_{eid[:8]}", "site": site, "event_id": eid, "time": ts,
                     "track": tid, "h": ph, "items": "+".join(sorted(items)), "crop": c})
print(f"{len(rows)} candidate crops after dedup ({a.per_track} per track, {a.gap:.0f}s apart)")

P = []
for i in range(0, len(rows), 128):
    x = np.stack([prep(r["crop"]) for r in rows[i:i+128]])
    with torch.no_grad(): P.append(torch.sigmoid(net(torch.from_numpy(x).cuda())).cpu().numpy())
P = np.concatenate(P) if P else np.zeros((0, 6))

out, n = [], collections.Counter()
for r, p in zip(rows, P):
    for it in r["items"].split("+"):
        k = ITEMS.index(it)
        name = f"{r['id']}.jpg"
        cv2.imwrite(str(OUT/it/name), r["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        n[(it, "agree" if p[k] >= 0.5 else "disagree")] += 1
        out.append({"item": it, "file": f"{it}/{name}", "site": r["site"], "event_id": r["event_id"],
                    "time": r["time"], "box_h": r["h"], "device": "ok",
                    "v5_p": round(float(p[k]), 3), "v5_k": round(float(p[3+k]), 3),
                    "verdict": ""})
with open(OUT/"candidates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(f"{len(out)} weak positives -> {OUT}/candidates.csv")
for it in ITEMS:
    ag, dis = n[(it, "agree")], n[(it, "disagree")]
    print(f"   {it:9} {ag+dis:5d}   v5 agrees (p>=0.5) {ag:5d}   v5 disagrees {dis:5d}")
