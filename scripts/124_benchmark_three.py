"""Three-way benchmark on the device camera's held-out clips: person model vs new packed vs production.

Systems, each run the way the device would run it, on people found by
yolo11s-pose @ 0.40 in the original frame:

  PERSON (new)      one network on the person crop (+18% pad, letterbox 320)
                    runs/person_ppe/efficientnet_b0_320_rec2
  PACKED (new)      the retrained classifier stack: head_cap + mask classifiers
                    chosen on VAL from runs/retrain_sppg, and the gloves crop
                    detector from runs/retrain_clean/gloves_det/yolo26n.
                    Same gates as the device: head gate for head_cap, head +
                    facing gate for mask.
  PACKED (deployed) the bundle as it runs today
  PRODUCTION        apd_mbg Dec-2025, its own detector on the whole frame; a
                    person's verdict comes from the boxes inside their person box

TEST = data/label_recording/test_split.csv (people from clips no model trained
or validated on). Scoring is operational: violation = item absent is positive,
"unknown" or a person the system misses counts as no alert, and people the
reviewer could not judge are skipped for that item.

    python scripts/124_benchmark_three.py
"""
import collections, csv, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
from scipy.stats import chi2
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd.pose import build_rois, face_roi, roi_usable
from apd.schema import GEOMETRY
from apd_perception_ours import _crop, _prep1, _facing_camera
from ultralytics import YOLO

ITEMS = ["head_cap", "mask", "gloves"]
NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = "cuda"

test = list(csv.DictReader(open(ROOT/"data/label_recording/test_split.csv")))
people = {r["person"]: r for r in csv.DictReader(open(ROOT/"data/label_recording/labels.csv"))}
pose = YOLO(str(ROOT/"yolo11s-pose.pt"))
prod = YOLO(str(ROOT/"artifacts/sdk/sdk/build/apd_mbg_openvino_model"), task="detect")


def torch_clf(spec, outputs=1):
    net = getattr(torchvision.models, spec["arch"])(weights=None)
    if hasattr(net, "fc"): net.fc = torch.nn.Linear(net.fc.in_features, outputs)
    else: net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, outputs)
    net.load_state_dict(torch.load(spec["weights"], map_location="cpu", weights_only=False))
    return net.eval().to(dev)


# ---- systems ---------------------------------------------------------------
pj = json.loads((ROOT/"runs/person_ppe/efficientnet_b0_320_rec2/summary.json").read_text())
person_net = torch_clf(pj, 3); person_th = {k: v["threshold"] for k, v in pj["items"].items()}

newp = {}
for at in ("head_cap", "mask"):
    s = json.loads((ROOT/f"runs/retrain_sppg/{at}/summary.json").read_text())
    best = max((k for k, v in s.items() if v["kind"] == "torch"), key=lambda k: s[k]["val_mcc"])
    newp[at] = (torch_clf(s[best]), s[best]["threshold"], best)
print("new packed picks:", {k: (v[2], v[1]) for k, v in newp.items()}, flush=True)
gl_new = YOLO(str(ROOT/"runs/retrain_clean/gloves_det/yolo26n/weights/best.pt"))
dep = {"head_cap": (torch_clf({"arch": "resnet34", "weights": ROOT/"runs/fixed_sdk/resnet34/head_cap_best.pt"}), 0.5),
       "mask": (torch_clf({"arch": "efficientnet_b0", "weights": ROOT/"runs/mask_face/g0.8_deployed/mask_best.pt"}), 0.30)}
gl_dep = YOLO(str(ROOT/"runs/det7/yolo26n/weights/best.pt"))


def letterbox(im, s=320):
    h, w = im.shape[:2]; r = s/max(h, w)
    im = cv2.resize(im, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = np.full((s, s, 3), 114, np.uint8); y, x = (s-im.shape[0])//2, (s-im.shape[1])//2
    o[y:y+im.shape[0], x:x+im.shape[1]] = im
    return o


@torch.no_grad()
def p1(net, crop, letter=False):
    x = (((cv2.cvtColor(letterbox(crop), cv2.COLOR_BGR2RGB).astype(np.float32)/255-MEAN)/STD).transpose(2, 0, 1)[None]
         if letter else _prep1(crop)[None])
    return torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(x)).to(dev))).cpu().numpy()[0]


def iou(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1])); i = ix*iy
    return i/max(1e-6, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)


def gloves_of(det, frame, pb):
    H, W = frame.shape[:2]; x1, y1, x2, y2 = pb; pw, ph = x2-x1, y2-y1
    c = frame[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
    if c.size == 0 or min(c.shape[:2]) < 16: return 0
    r = det.predict(c, imgsz=416, conf=0.25, verbose=False)[0]
    best = {}
    for cl, s in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        n = NAMES[int(cl)]
        if n in ("gloves", "no_gloves"): best[n] = max(best.get(n, 0), float(s))
    if not best: return 0
    return 1 if best.get("no_gloves", 0) > best.get("gloves", 0) else 0


SYS = ["PERSON effnet_b0 +rec2", "PACKED new (retrained)", "PACKED deployed", "PRODUCTION apd_mbg"]
Y = {it: [] for it in ITEMS}; Q = {s: {it: [] for it in ITEMS} for s in SYS}
cache = {}
for r in test:
    lab = people[r["image"]]
    key = (lab["clip"], lab["sec"])
    if key not in cache:
        v = next(ROOT.glob(f"data/recording/**/{lab['clip']}.mp4"))
        cap = cv2.VideoCapture(str(v)); cap.set(cv2.CAP_PROP_POS_MSEC, float(lab["sec"])*1000)
        ok, fr = cap.read(); cache = {key: fr if ok else None}
    frame = cache[key]
    if frame is None: continue
    box = json.loads(lab["box"])
    rr = pose.predict(frame, imgsz=640, conf=0.40, verbose=False)[0]
    B = rr.boxes.xyxy.cpu().numpy() if len(rr.boxes) else np.zeros((0, 4))
    K = rr.keypoints.data.cpu().numpy() if len(rr.boxes) else np.zeros((0, 17, 3))
    ppl = build_rois(B, K) if len(B) else []
    j = int(np.argmax([iou(p.person_box, box) for p in ppl])) if ppl else -1
    hit = j >= 0 and iou(ppl[j].person_box, box) >= 0.5
    # production: its own detector on the whole frame
    pr = prod.predict(frame, imgsz=640, conf=0.25, iou=0.45, agnostic_nms=True, verbose=False)[0]
    pb_boxes = [(prod.names[int(c)], b) for c, b in zip(pr.boxes.cls.cpu().numpy(), pr.boxes.xyxy.cpu().numpy())]
    inside = collections.Counter(n for n, b in pb_boxes
                                if box[0] <= (b[0]+b[2])/2 <= box[2] and box[1] <= (b[1]+b[3])/2 <= box[3])
    for it in ITEMS:
        y = int(r[it])
        if y < 0: continue
        Y[it].append(1 if y == 0 else 0)                      # violation positive
        # production
        Q["PRODUCTION apd_mbg"][it].append(1 if inside.get(f"no_{it}", 0) and not inside.get(it, 0) else 0)
        if not hit:
            for s in SYS[:3]: Q[s][it].append(0)
            continue
        pp = ppl[j]
        if it == "gloves":
            Q["PERSON effnet_b0 +rec2"][it].append(0)          # filled below, one call for all items
            Q["PACKED new (retrained)"][it].append(gloves_of(gl_new, frame, pp.person_box))
            Q["PACKED deployed"][it].append(gloves_of(gl_dep, frame, pp.person_box))
        else:
            ok_head = (pp.head is not None and not pp.head_collision and
                       roi_usable(pp.head, frame.shape, part="head", others=[q.person_box for q in ppl], self_box=pp.person_box))
            crop = _crop(frame, pp.head, GEOMETRY.head_crop_px) if ok_head else None
            if it == "mask" and crop is not None:
                fb = face_roi(K[j], tuple(B[j]), gain=0.8) if _facing_camera(K[j], 0.5, True) else None
                crop = None if fb is None else _crop(frame, fb, GEOMETRY.face_crop_px)
            for s, (net, th) in (("PACKED new (retrained)", newp[it][:2]), ("PACKED deployed", dep[it])):
                Q[s][it].append(0 if crop is None else int(float(p1(net, crop)[0]) < th))
            Q["PERSON effnet_b0 +rec2"][it].append(0)          # filled below
    # person model: one call for all three items
    if hit:
        H, W = frame.shape[:2]; x1, y1, x2, y2 = ppl[j].person_box; pw, ph = x2-x1, y2-y1
        c = frame[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
        p = p1(person_net, c, True) if c.size and min(c.shape[:2]) >= 16 else np.ones(3)
        for k, it in enumerate(ITEMS):
            if int(r[it]) < 0: continue
            Q["PERSON effnet_b0 +rec2"][it][-1] = int(float(p[k]) < person_th[it])


def mcnemar(y, a, b):
    ao, bo = a == y, b == y
    n, m = int((ao & ~bo).sum()), int((~ao & bo).sum())
    return 1.0 if n+m == 0 else float(chi2.sf((abs(n-m)-1)**2/(n+m), 1))


print(f"\nTEST = device camera, held-out clips ({len(test)} people)\n")
res = {}
for it in ITEMS:
    y = np.array(Y[it])
    print(f"{it}  ({len(y)} judged, {int(y.sum())} violations)")
    print(f"  {'system':28} {'prec':>7} {'recall':>7} {'MCC':>7}  {'p vs production':>16}")
    base = np.array(Q["PRODUCTION apd_mbg"][it])
    rows = []
    for s in SYS:
        q = np.array(Q[s][it])
        rows.append((s, precision_score(y, q, zero_division=0), recall_score(y, q, zero_division=0),
                     matthews_corrcoef(y, q) if len(set(q.tolist())) > 1 else 0.0,
                     "baseline" if s.startswith("PRODUCTION") else f"{mcnemar(y, base, q):.3f}"))
    for n, p, rc, m, pv in sorted(rows, key=lambda r: -r[3]):
        print(f"  {n:28} {p:7.1%} {rc:7.1%} {m:7.3f}  {pv:>16}")
    print()
    res[it] = [dict(zip(["system", "precision", "recall", "mcc", "p_vs_production"], r)) for r in rows]
(ROOT/"runs/person_ppe/benchmark_device_camera.json").write_text(json.dumps(res, indent=2))
