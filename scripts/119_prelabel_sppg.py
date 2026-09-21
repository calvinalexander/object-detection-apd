"""Model-assisted labelling, step 1: pick 3000 unlabelled people from data/sppg and pre-label them.

Selection
  - people from artifacts/persons.jsonl (pose boxes + keypoints, 11,305 people)
  - never a COCO test frame, and never any frame from a COCO test SCENE (scene_id),
    so near-duplicates of the test set cannot leak into training
  - person box at least 80 px tall, pose conf >= 0.40
  - spread over scenes: round-robin across scene_id, random within a scene

Pre-labels: the deployed bundle's models, on the geometry the device uses
  head_cap  resnet34 (runs/fixed_sdk)          head ROI, thr 0.50
  mask      effnet_b0 face crop (g0.8)         face ROI, thr 0.30, facing gate
  gloves    yolo26n crop detector (det7)       person box +18%, conf 0.25
Each item also gets "unknown" when the device would not judge it.

Output data/label_sppg/
  people.csv          id, frame, scene, person box, pre-label + score per item
  person/<id>.jpg     person crop (+18%), what a reviewer and the person model see
  sheets/sheet_NNN.jpg  audit sheets: person crop, zoomed head, zoomed hands, pre-labels

    python scripts/119_prelabel_sppg.py [--n 3000]
"""
import argparse, collections, csv, json, random, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd.pose import build_rois, face_roi, roi_usable
from apd.schema import GEOMETRY
from apd_perception_ours import _crop, _prep1, _facing_camera
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--min-h", type=int, default=80)
ap.add_argument("--out", default="data/label_sppg")
a = ap.parse_args()
OUT = ROOT/a.out; (OUT/"person").mkdir(parents=True, exist_ok=True); (OUT/"sheets").mkdir(exist_ok=True)
NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
rng = random.Random(20260917)

# ------------------------------------------------------------------ selection
P = [json.loads(l) for l in open(ROOT/"artifacts/persons.jsonl")]
coco_frames = {json.loads(l)["person_id"].split("#")[0] + ".jpg" for l in open(ROOT/"artifacts/seed_labels.jsonl")}
coco_scenes = {p["scene_id"] for p in P if p["frame"] in coco_frames}
pool = [p for p in P if p["frame"] not in coco_frames and p["scene_id"] not in coco_scenes
        and p["person_box"][3]-p["person_box"][1] >= a.min_h and p["person_conf"] >= 0.40]
by_scene = collections.defaultdict(list)
for p in pool: by_scene[p["scene_id"]].append(p)
for v in by_scene.values(): rng.shuffle(v)
order = sorted(by_scene); rng.shuffle(order)
pick = []
while len(pick) < a.n and any(by_scene.values()):
    for s in order:
        if by_scene[s] and len(pick) < a.n: pick.append(by_scene[s].pop())
print(f"persons {len(P)}; COCO frames {len(coco_frames)} in {len(coco_scenes)} scenes excluded; "
      f"eligible {len(pool)} in {len(by_scene)} scenes; picked {len(pick)}", flush=True)

# ------------------------------------------------------------------ models
def tnet(arch, w):
    n = getattr(torchvision.models, arch)(weights=None)
    if hasattr(n, "fc"): n.fc = torch.nn.Linear(n.fc.in_features, 1)
    else: n.classifier[-1] = torch.nn.Linear(n.classifier[-1].in_features, 1)
    n.load_state_dict(torch.load(w, map_location="cpu", weights_only=False)); return n.eval().cuda()


hc = tnet("resnet34", ROOT/"runs/fixed_sdk/resnet34/head_cap_best.pt")
mk = tnet("efficientnet_b0", ROOT/"runs/mask_face/g0.8_deployed/mask_best.pt")
gd = YOLO(str(ROOT/"runs/det7/yolo26n/weights/best.pt"))


@torch.no_grad()
def p1(net, c): return float(torch.sigmoid(net(torch.from_numpy(_prep1(c))[None].cuda())[0, 0]))


# ------------------------------------------------------------------ run
by_frame = collections.defaultdict(list)
for p in pick: by_frame[p["frame"]].append(p)
rows, tiles = [], []
T = 220
for fn, ppl in sorted(by_frame.items()):
    im = cv2.imread(str(ROOT/"data/sppg"/fn)); H, W = im.shape[:2]
    everyone = [q for q in P if q["frame"] == fn]                  # all people, for the head gate's "others"
    B = np.array([q["person_box"] for q in everyone], float)
    K = np.array([q["kp"] for q in everyone], float)
    rois = build_rois(B, K)
    for p in ppl:
        i = everyone.index(p); pp = rois[i]; pb = pp.person_box
        rec = {"id": p["person_id"].replace("#", "_p"), "frame": fn, "scene": p["scene_id"],
               "box": json.dumps([round(float(v), 1) for v in pb])}
        # head_cap + mask
        head_ok = (pp.head is not None and not pp.head_collision and
                   roi_usable(pp.head, im.shape, part="head", others=[r.person_box for r in rois], self_box=pb))
        hcrop = _crop(im, pp.head, GEOMETRY.head_crop_px) if head_ok else None
        if hcrop is None: rec.update(head_cap="unknown", head_cap_p="")
        else:
            s = p1(hc, hcrop); rec.update(head_cap="present" if s >= 0.5 else "absent", head_cap_p=round(s, 3))
        fb = face_roi(K[i], tuple(pb), gain=0.8) if head_ok and _facing_camera(K[i], 0.5, True) else None
        fcrop = None if fb is None else _crop(im, fb, GEOMETRY.face_crop_px)
        if fcrop is None: rec.update(mask="unknown", mask_p="")
        else:
            s = p1(mk, fcrop); rec.update(mask="present" if s >= 0.30 else "absent", mask_p=round(s, 3))
        # gloves
        x1, y1, x2, y2 = pb; pw, ph = x2-x1, y2-y1
        cx1, cy1, cx2, cy2 = int(max(0, x1-.18*pw)), int(max(0, y1-.18*ph)), int(min(W, x2+.18*pw)), int(min(H, y2+.18*ph))
        pcrop = im[cy1:cy2, cx1:cx2]
        best, gbox = {}, []
        rr = gd.predict(pcrop, imgsz=416, conf=0.25, verbose=False)[0]
        for c, sc, bb in zip(rr.boxes.cls.cpu().numpy(), rr.boxes.conf.cpu().numpy(), rr.boxes.xyxy.cpu().numpy()):
            n = NAMES[int(c)]
            if n in ("gloves", "no_gloves"):
                best[n] = max(best.get(n, 0), float(sc)); gbox.append((n, bb))
        if not best: rec.update(gloves="unknown", gloves_p="")
        else:
            v = "absent" if best.get("no_gloves", 0) > best.get("gloves", 0) else "present"
            rec.update(gloves=v, gloves_p=round(max(best.values()), 3))
        cv2.imwrite(str(OUT/"person"/f"{rec['id']}.jpg"), pcrop)
        rows.append(rec)

        # ---- audit tile: person | head zoom | hand zooms
        pc = pcrop.copy()
        for n, bb in gbox:
            cv2.rectangle(pc, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 170, 0) if n == "gloves" else (0, 0, 230), 2)
        s = T/pc.shape[0]; pc = cv2.resize(pc, (max(1, int(pc.shape[1]*s)), T))
        pc = pc[:, :150] if pc.shape[1] > 150 else cv2.copyMakeBorder(pc, 0, 0, 0, 150-pc.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        def zoom(box, size):
            if box is None: return np.full((size, size, 3), 235, np.uint8)
            bx1, by1, bx2, by2 = box; m = 0.25*max(bx2-bx1, by2-by1)
            z = im[int(max(0, by1-m)):int(min(H, by2+m)), int(max(0, bx1-m)):int(min(W, bx2+m))]
            return np.full((size, size, 3), 235, np.uint8) if z.size == 0 else cv2.resize(z, (size, size), interpolation=cv2.INTER_LANCZOS4)
        head_box = pp.head if pp.head is not None else None
        hz = zoom(head_box, T//2)
        hands = [((bb[0]+cx1, bb[1]+cy1, bb[2]+cx1, bb[3]+cy1)) for _, bb in gbox][:1] or list(pp.hands[:1]) or [None]
        gz = zoom(hands[0], T//2)
        right = np.vstack([hz, gz])
        t = cv2.copyMakeBorder(np.hstack([pc, right]), 44, 4, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        k = len(rows)-1
        col = lambda v: (0, 150, 0) if v == "present" else (0, 0, 220) if v == "absent" else (130, 130, 130)
        short = {"present": "yes", "absent": "NO", "unknown": "?"}
        cv2.putText(t, f"#{k}", (3, 16), 0, 0.55, (0, 0, 0), 2)
        cv2.putText(t, f"cap {short[rec['head_cap']]}", (60, 16), 0, 0.5, col(rec["head_cap"]), 2)
        cv2.putText(t, f"mask {short[rec['mask']]}", (160, 16), 0, 0.5, col(rec["mask"]), 2)
        cv2.putText(t, f"glv {short[rec['gloves']]}", (60, 36), 0, 0.5, col(rec["gloves"]), 2)
        tiles.append(t)
    if len(rows) % 500 < len(ppl): print(f"  {len(rows)}/{len(pick)}", flush=True)

for k, r in enumerate(rows): r["n"] = k
cols = ["n", "id", "frame", "scene", "box", "head_cap", "head_cap_p", "mask", "mask_p", "gloves", "gloves_p"]
with open(OUT/"people.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c: r[c] for c in cols})
PER, COLS = 24, 6
for s in range(0, len(tiles), PER):
    g = tiles[s:s+PER]; g += [np.full_like(tiles[0], 255)]*(PER-len(g))
    cv2.imwrite(str(OUT/"sheets"/f"sheet_{s//PER:03d}.jpg"), np.vstack([np.hstack(g[j:j+COLS]) for j in range(0, PER, COLS)]),
                [cv2.IMWRITE_JPEG_QUALITY, 93])
c = collections.Counter((it, r[it]) for r in rows for it in ("head_cap", "mask", "gloves"))
print("pre-labels", dict(sorted(c.items())), f"\nsheets {(len(tiles)+PER-1)//PER}  -> {OUT}")
