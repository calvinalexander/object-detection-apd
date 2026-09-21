"""TEST on the device camera: the held-out clips of data/label_recording.

data/label_recording/test_split.csv is written by scripts/118 --rec: people from
20% of the clips, never trained or validated on. Scores every person model in
runs/person_ppe/ plus the deployed stack's own pre-labels (people.csv), the same
operational way as the COCO test: violation = item absent is positive, a person
the reviewer could not judge is skipped for that item.

    python scripts/123_eval_recording_test.py
"""
import csv, json
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ["head_cap", "mask", "gloves"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
test = list(csv.DictReader(open(ROOT/"data/label_recording/test_split.csv")))
pre = {r["person"]: r for r in csv.DictReader(open(ROOT/"data/label_recording/labels.csv"))}


def letterbox(im, s):
    h, w = im.shape[:2]; r = s/max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return o


rows = []
for sp in sorted((ROOT/"runs/person_ppe").glob("*/summary.json")):
    j = json.loads(sp.read_text())
    net = getattr(torchvision.models, j["arch"])(weights=None)
    if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, 3)
    else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 3)
    net.load_state_dict(torch.load(j["weights"], map_location="cpu", weights_only=False))
    net = net.eval().cuda()
    X = np.stack([((cv2.cvtColor(letterbox(cv2.imread(str(ROOT/r["image"])), j["px"]), cv2.COLOR_BGR2RGB).astype(np.float32)/255-MEAN)/STD).transpose(2, 0, 1) for r in test])
    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(X).cuda())).cpu().numpy()
    name = f"{j['arch']}_{j['px']}" + (f" +{j['tag']}" if j.get("tag") else "")
    # a model trained with data/label_recording under a DIFFERENT split has seen
    # these clips: its score here is leakage, not a test
    if "label_recording" in (j.get("data") or "") and j.get("tag") != "rec2": name += "  [LEAKED]"
    rows.append((name, {it: (P[:, k], j["items"][it]["threshold"]) for k, it in enumerate(ITEMS)}))
# deployed stack, as it pre-labelled these same people
dep = {it: (np.array([1.0 if pre[r["image"]]["pre_" + it] == "present" else 0.0 for r in test]), 0.5) for it in ITEMS}
rows.insert(0, ("DEPLOYED stack (pre-labels)", dep))

print(f"TEST = device camera, held-out clips: {len(test)} people\n")
for it in ITEMS:
    y = np.array([r[it] for r in test], int)          # 1 present, 0 absent, -1 unclear
    m = y >= 0
    print(f"{it}  ({m.sum()} judged, {int((y[m] == 0).sum())} violations)")
    print(f"  {'model':34} {'thr':>5} {'prec':>7} {'recall':>7} {'MCC':>7}")
    out = []
    for name, d in rows:
        p, t = d[it]; q = (p[m] < t).astype(int); yy = (y[m] == 0).astype(int)
        out.append((name, t, precision_score(yy, q, zero_division=0), recall_score(yy, q, zero_division=0),
                    matthews_corrcoef(yy, q) if len(set(q.tolist())) > 1 else 0.0))
    for n, t, pr, rc, mc in sorted(out, key=lambda r: -r[4]):
        print(f"  {n:34} {t:5.2f} {pr:7.1%} {rc:7.1%} {mc:7.3f}")
    print()
