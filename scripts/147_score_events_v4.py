"""Score production's device events with a person-PPE checkpoint (default v4).

Reads the event database and the device's own snapshots, crops production's
person box (+18% pad, letterbox 320) and writes the six raw scores per event.

    python scripts/147_score_events_v4.py <out.json> [--run runs/person_ppe_v4/efficientnet_b0_320]
                                          [--arm artifacts/replay16_person/prod_device] [--day 2026-09-16]
"""
import argparse, json, sqlite3
from pathlib import Path
import cv2, numpy as np, torch, torchvision

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--run", default="runs/person_ppe_v4/efficientnet_b0_320")
ap.add_argument("--arm", default="artifacts/replay16_person/prod_device")
ap.add_argument("--day", default="2026-09-16")
a = ap.parse_args()
ITEMS = ["head_cap", "mask", "gloves"]
NAME = {"headcap": "head_cap", "mask": "mask", "gloves": "gloves"}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

j = json.loads((ROOT/a.run/"summary.json").read_text())
net = getattr(torchvision.models, j["arch"])(weights=None)
if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, 6)
else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 6)
net.load_state_dict(torch.load(ROOT/a.run/"best.pt", map_location="cpu", weights_only=False))
net = net.eval().cuda()
print(f"{a.run}: best epoch {j['best_epoch']}, val mean {j['val_mean']}")

ARM = ROOT/a.arm
db = sqlite3.connect(f"file:{ARM}/apd_storage/apd_events.db?mode=ro", uri=True)
rows = list(db.execute("select event_id,timestamp,camera_id,person_bbox,violation,image_path from events "
                       f"where timestamp like '{a.day}%'"))
print(f"{len(rows)} events")


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


for eid, ts, cam, bbox, vio, ip in rows:
    im = cv2.imread(str(ARM/ip))
    if im is None: miss += 1; continue
    H, W = im.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in json.loads(bbox)]
    pw, ph = x2-x1, y2-y1
    c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
    if c.size == 0 or min(c.shape[:2]) < 8: miss += 1; continue
    batch.append(prep(c))
    meta.append({"event": eid, "time": ts, "clip": cam, "vio": [NAME[x] for x in json.loads(vio)]})
    if len(batch) == 128: flush()
flush()
json.dump(out, open(a.out, "w"))
print(f"{len(out)} scored ({miss} skipped) -> {a.out}")
