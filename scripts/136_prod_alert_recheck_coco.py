"""Our person model as a second filter on PRODUCTION's own violation alerts, COCO labels.

Production (apd_mbg, its 7-class detector on the whole frame, conf 0.25) gives
both the person box and the alert. Every alert that lands on a human-labelled
person is re-judged by our model on production's own crop (+18% pad, letterbox
320), with that model's VAL cuts. The human label then says who was right.

    python scripts/136_prod_alert_recheck_coco.py [--runs runs/person_ppe_v2/efficientnet_b0_320,...]

NOTE v3 was trained with the COCO people folded in, so its numbers here are
optimistic; v2 never saw them.
"""
import argparse, collections, json
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/person_ppe_v2/efficientnet_b0_320,runs/person_ppe_v3/efficientnet_b0_320")
ap.add_argument("--out", default="artifacts/prod_alert_recheck_coco.json")
a = ap.parse_args()
ITEMS = ["head_cap", "mask", "gloves"]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

persons = {json.loads(l)["person_id"]: json.loads(l) for l in open(ROOT/"artifacts/persons.jsonl")}
truth = collections.defaultdict(dict)                      # item -> person_id -> present/absent/unclear
for l in open(ROOT/"artifacts/seed_labels.jsonl"):
    r = json.loads(l)
    if r["person_id"] in persons:
        for it, v in (r.get("labels") or {}).items(): truth[it][r["person_id"]] = v
frames = sorted({persons[p]["frame"] for it in truth for p in truth[it]})
prod = YOLO(str(ROOT/"artifacts/sdk/sdk/build/apd_mbg_openvino_model"), task="detect")


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix*iy
    return i / max(1e-6, (p[2]-p[0])*(p[3]-p[1]) + (q[2]-q[0])*(q[3]-q[1]) - i)


def lb(c, s=320):
    h, w = c.shape[:2]; r = s/max(h, w)
    im = cv2.resize(c, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return ((cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)


# ---- collect production's alerts on labelled people, with the crop it saw
alerts, crops = [], []
for fn in frames:
    im = cv2.imread(str(ROOT/"data/sppg"/fn))
    if im is None: continue
    H, W = im.shape[:2]
    r = prod.predict(im, imgsz=640, conf=0.25, iou=0.45, agnostic_nms=True, verbose=False)[0]
    nb = [(prod.names[int(c)], b) for c, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
    lab = [(pid, persons[pid]["person_box"]) for pid in persons if persons[pid]["frame"] == fn]
    for n, pb in nb:
        if n != "person": continue
        inside = collections.Counter(m for m, b in nb
                                     if pb[0] <= (b[0]+b[2])/2 <= pb[2] and pb[1] <= (b[1]+b[3])/2 <= pb[3])
        vio = [it for it in ITEMS if inside.get(f"no_{it}", 0) and not inside.get(it, 0)]
        if not vio: continue
        j = max(lab, key=lambda t: iou(pb, t[1])) if lab else None
        if j is None or iou(pb, j[1]) < 0.5: continue        # alert on nobody we labelled
        pid = j[0]
        vio = [it for it in vio if pid in truth[it]]         # only items a human looked at
        if not vio: continue
        x1, y1, x2, y2 = pb; pw, ph = x2-x1, y2-y1
        c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
        if c.size == 0 or min(c.shape[:2]) < 8: continue
        alerts.append({"frame": fn, "person_id": pid, "vio": vio,
                       "truth": {it: truth[it][pid] for it in vio}})
        crops.append(lb(c))
print(f"{len(alerts)} production alerts on labelled people, "
      f"{sum(len(a_['vio']) for a_ in alerts)} flagged items")

X = torch.from_numpy(np.stack(crops))
res = {}
for run in a.runs.split(","):
    j = json.loads((ROOT/run/"summary.json").read_text()); cuts = j["cuts"]
    net = getattr(torchvision.models, j["arch"])(weights=None)
    net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 6)
    net.load_state_dict(torch.load(ROOT/run/"best.pt", map_location="cpu", weights_only=False))
    net = net.eval().cuda()
    P = []
    with torch.no_grad():
        for i in range(0, len(X), 64): P.append(torch.sigmoid(net(X[i:i+64].cuda())).cpu().numpy())
    P = np.concatenate(P)
    name = Path(run).parents[0].name
    tab = {}
    for it in ITEMS:
        k = ITEMS.index(it); c = cuts[it]
        n = collections.Counter()
        for al, p in zip(alerts, P):
            if it not in al["vio"]: continue
            ours = ("unknown" if float(p[3+k]) < c["known_above"] else
                    "violation" if float(p[k]) < c["present_below"] else "ok")
            n[(al["truth"][it], ours)] += 1
        tab[it] = {f"{t}|{o}": v for (t, o), v in n.items()}
        T = lambda t: sum(v for (tt, _), v in n.items() if tt == t)
        K = lambda t: n[(t, "violation")]
        print(f"\n[{name}] {it}: {sum(n.values())} alerts  "
              f"(truth: {T('absent')} violation, {T('present')} compliant, {T('unclear')} unjudgeable)")
        print(f"   production RIGHT ({T('absent')}): kept {K('absent')}, "
              f"lost -> ok {n[('absent','ok')]}, abstain {n[('absent','unknown')]}")
        print(f"   production WRONG ({T('present')}): corrected -> ok {n[('present','ok')]}, "
              f"abstain {n[('present','unknown')]}; repeated {K('present')}")
        print(f"   unjudgeable ({T('unclear')}): abstained {n[('unclear','unknown')]}, "
              f"ok {n[('unclear','ok')]}, alerted {K('unclear')}")
    res[name] = tab
(ROOT/a.out).write_text(json.dumps({"alerts": alerts, "by_model": res}, indent=1))
print(f"\nwrote {a.out}")
