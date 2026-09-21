"""Pick both v2 thresholds per item on VALIDATION, for the decision that is actually made.

For each item, grid over (known_above, present_below) and score the operational
rule on val:  alert  <=>  p_known >= known_above AND p_present < present_below.
Objective: highest MCC on the people a human could judge, subject to the flag rate
on unjudgeable people staying at or below the deployed pipeline's COCO rate
(head_cap 27%, mask 14%, gloves 7%). COCO is not used here.

    python scripts/129_tune_v2_thresholds.py
Updates cuts in runs/person_ppe_v2/<tag>/summary.json (keeps the old ones as cuts_judge_only).
"""
import csv, hashlib, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ITEMS = ["head_cap", "mask", "gloves"]
CAP = {"head_cap": 0.27, "mask": 0.14, "gloves": 0.07}          # deployed pipeline's false-firing rate
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
sys.path.insert(0, str(ROOT/"scripts"))
from importlib import import_module
v2 = import_module("127_train_person_ppe_v2") if False else None


def lb(im, s=320):
    h, w = im.shape[:2]; r = s/max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return ((cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)


# val rows, same split as training
V = {"present": (1, 1), "absent": (0, 1), "unclear": (-1, 0)}
rows = []
for r in csv.DictReader(open(ROOT/"data/label_sppg/labels.csv")):
    if "not_person" in (r["head_cap"], r["mask"], r["gloves"]): continue
    if int(hashlib.md5(("s"+r["scene"]).encode()).hexdigest(), 16) % 100 >= 15: continue
    pres, known = zip(*[V[r[i]] for i in ITEMS])
    rows.append((r["person"], list(pres), list(known)))
D = ROOT/"data/clean/det7"
for lp in sorted((D/"labels/val").glob("*.txt")):
    cls = {int(l.split()[0]) for l in lp.read_text().splitlines() if l.strip()}
    pres, known = [], []
    for pos, neg in ((2, 5), (3, 6), (1, 4)):
        if neg in cls: pres.append(0); known.append(1)
        elif pos in cls: pres.append(1); known.append(1)
        else: pres.append(-1); known.append(-1)
    rows.append((str((D/"images/val"/(lp.stem+".jpg")).relative_to(ROOT)), pres, known))
print(f"{len(rows)} val people")

import os
for sp in sorted((ROOT/os.environ.get("PPE_RUNS","runs/person_ppe_v2")).glob("*/summary.json")):
    j = json.loads(sp.read_text())
    net = getattr(torchvision.models, j["arch"])(weights=None)
    if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, 6)
    else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 6)
    net.load_state_dict(torch.load(j["weights"], map_location="cpu")); net = net.eval().cuda()
    P = []
    with torch.no_grad():
        for i in range(0, len(rows), 128):
            x = np.stack([lb(cv2.imread(str(ROOT/p))) for p, _, _ in rows[i:i+128]])
            P.append(torch.sigmoid(net(torch.from_numpy(x).cuda())).cpu().numpy())
    P = np.concatenate(P)
    pres = np.array([r[1] for r in rows]); known = np.array([r[2] for r in rows])
    cuts = {}
    print(f"\n{j['arch']}")
    for k, it in enumerate(ITEMS):
        jud = pres[:, k] >= 0; unc = known[:, k] == 0
        Y = (pres[jud, k] == 0).astype(int)
        best = None
        for ka in np.arange(0.05, 0.96, 0.025):
            for pb in np.arange(0.05, 0.96, 0.025):
                q = ((P[jud, 3+k] >= ka) & (P[jud, k] < pb)).astype(int)
                if len(set(q)) < 2: continue
                m = matthews_corrcoef(Y, q)
                fr = float(np.mean((P[unc, 3+k] >= ka) & (P[unc, k] < pb))) if unc.sum() else 0.0
                if fr <= CAP[it] and (best is None or m > best[0]): best = (m, float(ka), float(pb), fr)
        if best is None: best = (0, 0.5, j["cuts"][it]["present_below"], 1.0)
        m, ka, pb, fr = best
        cuts[it] = {"known_above": round(ka, 3), "present_below": round(pb, 3),
                    "val_mcc_operational": round(m, 4), "val_unjudgeable_flagged": round(fr, 3)}
        print(f"  {it:9} known>={ka:.2f}  present<{pb:.2f}  val MCC {m:.3f}  unjudgeable flagged {fr:.0%}")
    j["cuts_judge_only"] = j["cuts"]; j["cuts"] = cuts
    j["rule"] = "alert when p_known >= known_above AND p_present < present_below (tuned jointly on val)"
    sp.write_text(json.dumps(j, indent=2))
