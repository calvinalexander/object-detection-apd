"""Deployed mask trunk against the retrained face-crop trunks, on held-out ground.

TEST: 172 frames from videos neither model trained on (script 101), faces as the
device sees them (yolo11n at 0.40, no jitter). Each new trunk uses the threshold
it picked on VAL; the deployed trunk uses 0.60, which was chosen on a set that
overlaps these videos, so the comparison flatters the deployed trunk slightly.

Second check: the 44 COCO human labels, run end to end through pose.
"""
import collections, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from scipy.stats import chi2
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd.pose import build_rois, face_roi
from apd.schema import GEOMETRY
from apd_perception_ours import _crop, _prep1
import onnxruntime as ort
from ultralytics import YOLO

summ = json.loads((ROOT/"runs/mask_face/summary.json").read_text())
packed = ort.InferenceSession(str(ROOT/"deploy/apd_ours/apd_ours_packed.onnx"), providers=["CPUExecutionProvider"])
OLD_T = 0.6


def load(tag):
    net = torchvision.models.efficientnet_b0(weights=None)
    net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
    net.load_state_dict(torch.load(ROOT/f"runs/mask_face/{tag}/mask_best.pt", map_location="cpu", weights_only=False))
    return net.eval().cuda()


nets = {t: load(t) for t in summ}


def geom_box(tag, kp, box, head):
    g = tag.split("_")[0]
    return head if g == "ghead" else face_roi(np.array(kp), tuple(box), gain=float(g[1:]))


@torch.no_grad()
def p_new(tag, im, kp, box, head):
    b = geom_box(tag, kp, box, head)
    c = None if b is None else _crop(im, b, GEOMETRY.face_crop_px)
    if c is None: return None
    return float(torch.sigmoid(nets[tag](torch.from_numpy(_prep1(c))[None].cuda())[0, 0]))


def p_old(im, head):
    c = _crop(im, head, GEOMETRY.head_crop_px)
    if c is None: return None
    x = _prep1(c)[None]
    return float(packed.run(None, {"head_crops": x, "hand_crops": np.zeros((0, 3, 128, 128), np.float32)})[1][0])


def mcnemar(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    return 1.0 if b + c == 0 else float(chi2.sf((abs(b-c)-1)**2/(b+c), 1))


def score(name, y, p, t):
    q = (p < t).astype(int)                      # violation = no mask = positive
    return (name, precision_score(y, q, zero_division=0), recall_score(y, q, zero_division=0),
            matthews_corrcoef(y, q) if len(set(q)) > 1 else 0.0, t, q)


def table(title, y, rows, old_q):
    print(f"\n{title}   n={len(y)}, no mask {int(y.sum())}, mask on {int((y==0).sum())}")
    print(f"  {'model':26} {'thresh':>6} {'prec':>7} {'recall':>7} {'MCC':>7}  {'wrongly flagged':>15}  p vs deployed")
    out = {}
    for name, pr, rc, m, t, q in rows:
        fp = int(((q == 1) & (y == 0)).sum())
        pv = "baseline" if q is old_q else f"{mcnemar(old_q == y, q == y):.1e}"
        print(f"  {name:26} {t:6.2f} {pr:7.1%} {rc:7.1%} {m:7.3f}  {fp:6d} of {int((y==0).sum()):<6d}  {pv}")
        out[name] = {"precision": pr, "recall": rc, "mcc": m, "threshold": t}
    return out


# ---------------------------------------------------------------- TEST
man = json.loads((ROOT/"data/mask_face/test_manifest.json").read_text())
Y, P = [], collections.defaultdict(list)
cache = {}
for m in man:
    im = cache.setdefault(m["img"], cv2.imread(m["img"]))
    vals = {"deployed (head crop)": p_old(im, m["head"])}
    for t in nets: vals[t] = p_new(t, im, m["kp"], m["box"], m["head"])
    if any(v is None for v in vals.values()): continue
    Y.append(0 if m["present"] else 1)
    for k, v in vals.items(): P[k].append(v)
Y = np.array(Y)
old = score("deployed (head crop)", Y, np.array(P["deployed (head crop)"]), OLD_T)
rows = [old] + [score(t, Y, np.array(P[t]), summ[t]["val_threshold"]) for t in nets]
rows.sort(key=lambda r: -r[3] if r is not old else 9)
res = {"test": table("TEST (held-out videos, device geometry)", Y, rows, old[5])}
best_tag = max(nets, key=lambda t: summ[t]["val_mcc"])
print(f"\nchosen on VAL: {best_tag} (val MCC {summ[best_tag]['val_mcc']:.3f}, threshold {summ[best_tag]['val_threshold']:.2f})")

# ---------------------------------------------------------------- keypoint noise
# The device runs yolo11n as int8 on the accelerator, so its keypoints are
# noisier than these float ones. A face crop is tighter than the head crop and
# should be the more sensitive of the two; this measures by how much.
print("\nkeypoint noise (sigma as a fraction of shoulder width), best-VAL model vs deployed, MCC:")
rng = np.random.default_rng(0)
from apd.schema import GEOMETRY as G
from apd.pose import head_roi
for f in (0.0, 0.03, 0.06, 0.10):
    yy, po, pn = [], [], []
    for m in man:
        kp = np.array(m["kp"], float)
        sh = abs(kp[5][0]-kp[6][0]) if min(kp[5][2], kp[6][2]) >= G.kp_visible else 0.25*(m["box"][2]-m["box"][0])
        kp[:, :2] += rng.normal(0, f*sh, (17, 2))
        im = cache[m["img"]]
        h = head_roi(kp, tuple(m["box"]))
        if h is None: continue
        a_, b_ = p_old(im, h), p_new(best_tag, im, kp, m["box"], h)
        if a_ is None or b_ is None: continue
        yy.append(0 if m["present"] else 1); po.append(a_); pn.append(b_)
    yy = np.array(yy)
    mo = matthews_corrcoef(yy, (np.array(po) < OLD_T).astype(int))
    mn = matthews_corrcoef(yy, (np.array(pn) < summ[best_tag]["val_threshold"]).astype(int))
    print(f"  noise {f:4.2f}:  deployed {mo:.3f}   {best_tag} {mn:.3f}   (n={len(yy)})")
    res.setdefault("noise", {})[f] = {"deployed": mo, best_tag: mn}

# ---------------------------------------------------------------- COCO human labels
persons = {json.loads(l)["person_id"]: json.loads(l) for l in open(ROOT/"artifacts/persons.jsonl")}
truth = {}
for l in open(ROOT/"artifacts/seed_labels.jsonl"):
    r = json.loads(l); v = (r.get("labels") or {}).get("mask")
    if r["person_id"] in persons and v in ("present", "absent"): truth[r["person_id"]] = v
pose = YOLO(str(ROOT/"yolo11n-pose.pt"))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix*iy
    return i / max(1e-6, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


Yc, Pc, Pn = [], [], []
for fn in sorted({persons[p]["frame"] for p in truth}):
    im = cv2.imread(str(ROOT/"data/sppg"/fn))
    r = pose.predict(im, imgsz=640, conf=0.40, verbose=False)[0]
    if not len(r.boxes): continue
    B = r.boxes.xyxy.cpu().numpy(); K = r.keypoints.data.cpu().numpy()
    ppl = build_rois(B, K)
    for pid, tv in truth.items():
        if persons[pid]["frame"] != fn: continue
        v = [iou(p.person_box, persons[pid]["person_box"]) for p in ppl]
        i = int(np.argmax(v))
        if v[i] < .3 or ppl[i].head is None: continue
        a_, b_ = p_old(im, ppl[i].head), p_new(best_tag, im, K[i], B[i], ppl[i].head)
        if a_ is None or b_ is None: continue
        Yc.append(1 if tv == "absent" else 0); Pc.append(a_); Pn.append(b_)
Yc = np.array(Yc)
o2 = score("deployed (head crop)", Yc, np.array(Pc), OLD_T)
res["coco"] = table("COCO human labels (end to end)", Yc,
                    [o2, score(best_tag, Yc, np.array(Pn), summ[best_tag]["val_threshold"])], o2[5])
res["chosen"] = best_tag
(ROOT/"runs/mask_face/eval.json").write_text(json.dumps(res, indent=2, default=float))
