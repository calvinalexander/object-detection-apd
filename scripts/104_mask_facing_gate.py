"""Stop judging mask on people who face away.

Today mask is judged whenever the head region passes the gate, so a back of a
head gets a mask verdict. Candidate gates, measured on two populations:

  refuse   COCO people whose mask a human marked UNCLEAR (mouth not visible), 255
  allow    faces a human could judge: COCO present/absent (44) and the
           annotator mask/no_mask faces from the held-out test videos (323)

A good gate passes the second and refuses the first.
"""
import collections, json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apd.pose import build_rois, frontality, roi_usable
from apd.schema import GEOMETRY as G
from ultralytics import YOLO

pose = YOLO(str(ROOT/"yolo11n-pose.pt"))
sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd_perception_ours import _crop, _prep1
import onnxruntime as ort
mask_net = ort.InferenceSession(str(ROOT/"deploy/apd_ours/apd_ours_mask.onnx"), providers=["CPUExecutionProvider"])
from sklearn.metrics import matthews_corrcoef
VIS = G.kp_visible


def features(kp, pp, pboxes, shape):
    head_ok = pp.head is not None and not pp.head_collision and \
        bool(roi_usable(pp.head, shape, part="head", others=pboxes, self_box=pp.person_box))
    sh = kp[5][2] >= VIS and kp[6][2] >= VIS
    ey = kp[1][2] >= VIS and kp[2][2] >= VIS
    return {"head_ok": head_ok, "front": frontality(kp[:7], VIS),
            # facing the camera, the person's LEFT side appears on the image's RIGHT
            # bool(): a numpy bool is never `is True` / `is False`
            "sh_facing": bool(kp[5][0] > kp[6][0]) if sh else None,
            "ear_facing": bool(kp[3][0] > kp[4][0]) if (kp[3][2] >= VIS and kp[4][2] >= VIS) else None,
            "nose": float(kp[0][2]), "eyes": float(min(kp[1][2], kp[2][2]))}


GATES = {
    "current (head gate only)": lambda f: f["head_ok"],
    "+ frontality >= 0.3": lambda f: f["head_ok"] and f["front"] >= 0.3,
    "+ shoulders not turned away": lambda f: f["head_ok"] and f["sh_facing"] is not False,
    "+ nose conf >= 0.5": lambda f: f["head_ok"] and f["nose"] >= 0.5,
    "+ nose conf >= 0.6": lambda f: f["head_ok"] and f["nose"] >= 0.6,
    "+ nose conf >= 0.7": lambda f: f["head_ok"] and f["nose"] >= 0.7,
    "+ nose >= 0.5 & eyes >= 0.35": lambda f: f["head_ok"] and f["nose"] >= 0.5 and f["eyes"] >= 0.35,
    "+ nose >= 0.5 & not turned away": lambda f: f["head_ok"] and f["nose"] >= 0.5
                                         and f["sh_facing"] is not False and f["ear_facing"] is not False,
    "+ nose >= 0.6 & not turned away": lambda f: f["head_ok"] and f["nose"] >= 0.6
                                         and f["sh_facing"] is not False and f["ear_facing"] is not False,
}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix*iy
    return i / max(1e-6, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


pop = collections.defaultdict(list)          # population -> [features]
examples = []                                # (population, image, box, features) for the sheet

# COCO, by label
persons = {json.loads(l)["person_id"]: json.loads(l) for l in open(ROOT/"artifacts/persons.jsonl")}
lab = {}
for l in open(ROOT/"artifacts/seed_labels.jsonl"):
    r = json.loads(l); v = (r.get("labels") or {}).get("mask")
    if r["person_id"] in persons and v: lab[r["person_id"]] = v
for fn in sorted({persons[p]["frame"] for p in lab}):
    im = cv2.imread(str(ROOT/"data/sppg"/fn))
    if im is None: continue
    r = pose.predict(im, imgsz=640, conf=0.40, verbose=False)[0]
    if not len(r.boxes): continue
    B = r.boxes.xyxy.cpu().numpy(); K = r.keypoints.data.cpu().numpy()
    ppl = build_rois(B, K); pbx = [p.person_box for p in ppl]
    for pid, v in lab.items():
        if persons[pid]["frame"] != fn: continue
        s = [iou(p.person_box, persons[pid]["person_box"]) for p in ppl]
        i = int(np.argmax(s))
        if s[i] < .3: continue
        f = features(K[i], ppl[i], pbx, im.shape)
        key = "refuse: COCO mask unclear" if v == "unclear" else "allow: COCO mask judged"
        pop[key].append(f)
        if v == "unclear" and f["head_ok"]: examples.append((str(ROOT/"data/sppg"/fn), ppl[i].person_box, f))

# held-out annotated faces
man = json.loads((ROOT/"data/mask_face/test_manifest.json").read_text())
cache = {}
for m in man:
    im = cache.setdefault(m["img"], cv2.imread(m["img"]))
    r = pose.predict(im, imgsz=640, conf=0.40, verbose=False)[0] if m["img"] not in cache.get("_p", {}) else None
    kp = np.array(m["kp"]); pp = build_rois(np.array([m["box"]]), kp[None])[0]
    f = features(kp, pp, [pp.person_box], im.shape)
    c = _crop(im, pp.face, G.face_crop_px) if pp.face is not None else None
    f["p_mask"] = None if c is None else float(mask_net.run(None, {"face_crops": _prep1(c)[None]})[0][0])
    f["truth_violation"] = not m["present"]
    pop["allow: annotated test faces"].append(f)

print(f"{'gate':34} " + " ".join(f"{k[:28]:>30}" for k in pop))
print(f"{'':34} " + " ".join(f"{'(n=' + str(len(v)) + ')':>30}" for v in pop.values()))
for g, fn in GATES.items():
    print(f"{g:34} " + " ".join(f"{np.mean([fn(f) for f in v]):29.0%} " for v in pop.values()))
print("\nmask model on the annotated test faces under each gate (abstain = no alert, threshold 0.30):")
T = [f for f in pop["allow: annotated test faces"] if f["p_mask"] is not None]
y = np.array([f["truth_violation"] for f in T]).astype(int)
for g, fn in GATES.items():
    q = np.array([int(fn(f) and f["p_mask"] < 0.30) for f in T])
    judged = np.array([fn(f) for f in T])
    mj = matthews_corrcoef(y[judged], q[judged]) if judged.sum() > 5 and len(set(q[judged])) > 1 else float("nan")
    print(f"  {g:34} operational MCC {matthews_corrcoef(y, q):.3f}   MCC on the faces it judges {mj:.3f}   judged {judged.mean():.0%}")
print("\n(the 'refuse' column is the share still judged -- lower is better;"
      " the 'allow' columns are the share still judged -- higher is better)")

# what the current gate lets through among the unclear ones, and what each cue says
out = ROOT/"artifacts/error_analysis/mask_facing"; out.mkdir(parents=True, exist_ok=True)
tiles = []
for img, box, f in examples[:24]:
    im = cv2.imread(img); H, W = im.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]; ph = y2-y1
    c = im[max(0, y1):min(H, y1+int(ph*.55)), max(0, x1):min(W, x2)]
    if c.size == 0: continue
    c = cv2.resize(c, (160, int(160*c.shape[0]/max(1, c.shape[1]))))[:200]
    t = np.full((230, 160, 3), 25, np.uint8); t[30:30+c.shape[0], :c.shape[1]] = c
    cv2.putText(t, f"front {f['front']:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .37, (230, 230, 230), 1)
    cv2.putText(t, f"shoulders {'facing' if f['sh_facing'] else 'AWAY' if f['sh_facing'] is False else '?'}",
                (3, 25), cv2.FONT_HERSHEY_SIMPLEX, .37, (160, 220, 160), 1)
    tiles.append(t)
while len(tiles) % 6: tiles.append(np.full((230, 160, 3), 25, np.uint8))
cv2.imwrite(str(out/"_unclear_but_judged_today.jpg"), np.vstack([np.hstack(tiles[i:i+6]) for i in range(0, len(tiles), 6)]))
print(f"\nwrote {out/'_unclear_but_judged_today.jpg'}")
