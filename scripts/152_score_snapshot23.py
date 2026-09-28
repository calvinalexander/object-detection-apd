"""Score the other site's events (data/snapshot/23) with a person-PPE checkpoint."""
import argparse, json, sqlite3
from pathlib import Path
import cv2, numpy as np, torch, torchvision

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("out")
ap.add_argument("--run", default="runs/person_ppe_v5/efficientnet_b0_320")
a = ap.parse_args()
SNAP = ROOT/"data/snapshot/23"
NAME = {"headcap": "head_cap", "mask": "mask", "gloves": "gloves"}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
j = json.loads((ROOT/a.run/"summary.json").read_text())
net = getattr(torchvision.models, j["arch"])(weights=None)
if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, 6)
else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 6)
net.load_state_dict(torch.load(ROOT/a.run/"best.pt", map_location="cpu", weights_only=False))
net = net.eval().cuda()
db = sqlite3.connect(f"file:{SNAP}/db?mode=ro", uri=True)
rows = list(db.execute("select event_id,timestamp,person_bbox,violation from events"))
det = {}
for e, p, s in db.execute("select event_id,ppe_type,status from apd_detail"): det.setdefault(e, {})[NAME.get(p, p)] = s


def prep(c, s=320):
    h, w = c.shape[:2]; r = s/max(h, w)
    im = cv2.resize(c, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return ((cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)


out, batch, meta, miss = [], [], [], 0
def flush():
    if not batch: return
    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(np.stack(batch)).cuda())).cpu().numpy()
    for p, m in zip(P, meta): out.append({**m, "p": [round(float(x), 4) for x in p]})
    batch.clear(); meta.clear()

for eid, ts, bbox, vio in rows:
    f = SNAP/f"{eid}.jpg"
    im = cv2.imread(str(f)) if f.exists() else None
    if im is None: miss += 1; continue
    H, W = im.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in json.loads(bbox)]
    pw, ph = x2-x1, y2-y1
    c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
    if c.size == 0 or min(c.shape[:2]) < 8: miss += 1; continue
    batch.append(prep(c))
    meta.append({"event": eid, "time": ts, "vio": [NAME[x] for x in json.loads(vio)], "detail": det.get(eid, {}), "h": ph})
    if len(batch) == 128: flush()
flush()
json.dump(out, open(a.out, "w"))
print(f"{len(out)} scored ({miss} skipped) -> {a.out}")
