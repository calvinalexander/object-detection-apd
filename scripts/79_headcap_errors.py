"""Why is head_cap only ~0.47 MCC? Look at what it gets wrong.

Two candidate explanations are testable without opening a single image:
whether errors concentrate on small/occluded/clipped ROIs (a crop-quality
problem), and whether they concentrate on one direction (a rule-definition
problem). The rest needs eyes, so the false positives and false negatives are
rendered for review.
"""
import collections, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.pose import roi_usable
from apd.schema import GEOMETRY, assert_contract

assert_contract()
dev = "cuda"
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
net = torchvision.models.resnet34(weights=None)
net.fc = torch.nn.Linear(net.fc.in_features, 1)
net.load_state_dict(torch.load("runs/bakeoff_torch/resnet34/head_cap_best.pt", map_location="cpu", weights_only=False))
net = net.to(dev).eval()

persons = {json.loads(l)["person_id"]: json.loads(l) for l in Path("artifacts/persons.jsonl").read_text().splitlines()}
truth = {}
for line in Path("artifacts/seed_labels.jsonl").read_text().splitlines():
    r = json.loads(line)
    v = (r.get("labels") or {}).get("head_cap")
    if v in ("present","absent") and r["person_id"] in persons:
        truth[r["person_id"]] = v

def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

cells = collections.defaultdict(list)
feat = []
for pid, tv in truth.items():
    rec = persons[pid]
    im = cv2.imread(str(Path("data/sppg")/rec["frame"]))
    if im is None or not rec.get("head_roi"): continue
    hc = crop_np(im, rec["head_roi"], GEOMETRY.head_crop_px)
    if hc is None: continue
    with torch.no_grad():
        x = (cv2.cvtColor(cv2.resize(hc,(128,128)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
        p = float(torch.sigmoid(net(torch.tensor(x.transpose(2,0,1)[None]).to(dev))[:,0])[0])
    pred = "present" if p >= .5 else "absent"
    kind = {("present","present"):"TP_present", ("absent","absent"):"TN_absent",
            ("absent","present"):"FP_saidCap", ("present","absent"):"FN_missedCap"}[(tv,pred)]
    hb = rec["head_roi"]
    side = min(hb[2]-hb[0], hb[3]-hb[1])
    ck = roi_usable(hb, im.shape, part="head", others=[], self_box=rec["person_box"])
    cells[kind].append((pid, p, hc))
    feat.append(dict(kind=kind, side=side, p=p, conf=rec.get("person_conf",0),
                     frontal=rec.get("frontal_face"), collide=rec.get("head_collision"),
                     gated=not bool(ck)))

n = {k: len(v) for k, v in cells.items()}
tot = sum(n.values())
print(f"{tot} persons with a head label and a usable crop\n")
print(f"{'outcome':16} {'n':>5} {'share':>7} {'median ROI px':>14} {'median |p-0.5|':>15}")
for k in ("TP_present","TN_absent","FP_saidCap","FN_missedCap"):
    f = [x for x in feat if x["kind"] == k]
    if not f: continue
    print(f"{k:16} {len(f):5d} {len(f)/tot:7.1%} {np.median([x['side'] for x in f]):14.0f} "
          f"{np.median([abs(x['p']-0.5) for x in f]):15.3f}")

print("\nis it crop quality? (compare errors vs correct)")
err = [x for x in feat if x["kind"].startswith(("FP","FN"))]
ok  = [x for x in feat if x["kind"].startswith(("TP","TN"))]
for label, sel in (("correct", ok), ("errors", err)):
    print(f"  {label:8} n={len(sel):4d}  median ROI {np.median([x['side'] for x in sel]):5.0f}px  "
          f"head_collision {sum(x['collide'] for x in sel)/len(sel):5.1%}  "
          f"gated-out {sum(x['gated'] for x in sel)/len(sel):5.1%}  "
          f"frontal {sum(bool(x['frontal']) for x in sel)/len(sel):5.1%}")

# render the errors
for k in ("FP_saidCap","FN_missedCap"):
    v = sorted(cells[k], key=lambda t: -abs(t[1]-0.5))[:12]
    if not v: continue
    tiles = []
    for pid, p, hc in v:
        t = cv2.resize(hc, (150,150))
        cv2.rectangle(t, (0,0), (149,149), (0,0,220) if k.startswith("FP") else (0,140,255), 3)
        cv2.putText(t, f"{p:.2f}", (5,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        tiles.append(t)
    while len(tiles) % 6: tiles.append(np.full((150,150,3),240,np.uint8))
    rows = [np.concatenate(tiles[i:i+6],axis=1) for i in range(0,len(tiles),6)]
    cv2.imwrite(f"artifacts/_headcap_{k}.jpg", np.concatenate(rows,axis=0))
    print(f"wrote artifacts/_headcap_{k}.jpg  ({len(cells[k])} cases)")
