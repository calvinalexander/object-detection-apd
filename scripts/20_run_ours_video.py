"""Run OUR pose-anchored pipeline over a video.

pose -> anatomical ROI -> per-part classifier, which is the design this project
adopted instead of full-frame detection. Nothing is orphaned here: PPE is defined
relative to a person, so there is no assignment step to lose detections in.

Classifiers are linear probes on frozen DINOv2 features, fit on the TRAIN split
only. The demo video is a camera none of it has seen, which makes this the
cross-camera evidence the frozen splits are too small to provide.
"""
import argparse, collections, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
from apd.pose import build_rois, frontality
from apd.schema import Attribute, Value, GEOMETRY

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", default="artifacts/ours_demo.mp4")
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--pose-weights", default=".work/yolo11x-pose.pt")
ap.add_argument("--frontality", type=float, default=0.5)
args = ap.parse_args()

DEVICE, MODEL_ID = "cuda", "facebook/dinov2-base"
CROPS = Path("artifacts/crops")
OK, BAD, UNK = (70, 190, 120), (60, 60, 235), (150, 155, 160)

proc = AutoImageProcessor.from_pretrained(MODEL_ID)
enc = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

@torch.no_grad()
def embed_pil(ims, batch=64):
    out = []
    for i in range(0, len(ims), batch):
        px = proc(images=ims[i:i+batch], return_tensors="pt").to(DEVICE)
        out.append(enc(**px).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

# ---- fit the probes on the train split -----------------------------------
splits = json.loads(Path("artifacts/splits.json").read_text())["splits"]
fs = {f: k for k, v in splits.items() for f in v}
index = [json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]
head_crop = {r["person_id"]: r["crop"] for r in index if r["part"] == "head"}
hand_by = {}
for r in index:
    if r["part"] == "hand":
        hand_by[(r["person_id"], int(r["crop"].rsplit("_h",1)[1].split(".")[0]))] = r["crop"]
seed = [json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()]
mined = [m for m in (json.loads(l) for l in Path("artifacts/mined_labels.jsonl").read_text().splitlines())
         if m["attribute"] == "mask"]
bulk = [json.loads(l) for l in Path("artifacts/bulk_labels.jsonl").read_text().splitlines()]

def fit(paths, ys, name):
    X = embed_pil([Image.open(CROPS/p).convert("RGB") for p in paths])
    y = np.array(ys)
    clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(X, y)
    print(f"  {name:<10} {int(y.sum())} present / {int(len(y)-y.sum())} absent")
    return clf

print("fitting probes on the train split")
rows = {a: ([], []) for a in ("head_cap","mask","gloves")}
for r in seed:
    if fs.get(r["frame"]) != "train": continue
    for a in ("head_cap","mask"):
        v = r["labels"][a]
        if v != Value.UNCLEAR.value and r["person_id"] in head_crop:
            rows[a][0].append(head_crop[r["person_id"]]); rows[a][1].append(int(v=="present"))
    for h in r.get("hands", []):
        c = hand_by.get((r["person_id"], h["hand"]))
        if c and h["value"] != Value.UNCLEAR.value:
            rows["gloves"][0].append(c); rows["gloves"][1].append(int(h["value"]=="present"))
for m in mined + [b for b in bulk if b.get("attribute")=="mask"]:
    if m.get("split")=="train" and m["value"]!="unclear" and m["person_id"] in head_crop:
        rows["mask"][0].append(head_crop[m["person_id"]]); rows["mask"][1].append(int(m["value"]=="present"))
clf = {a: fit(p, y, a) for a,(p,y) in rows.items()}

# ---- run the video --------------------------------------------------------
pose = YOLO(args.pose_weights)
cap = cv2.VideoCapture(args.video)
W,H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
wr = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS/args.stride, (W,H))
print(f"\nvideo: {W}x{H}, {total} frames -> {args.out}")

def crop_of(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    if c.size == 0 or c.shape[0] < 8 or c.shape[1] < 8: return None
    return Image.fromarray(cv2.resize(c, (size,size), interpolation=cv2.INTER_LANCZOS4))

stats = collections.Counter(); npersons = []; n = 0; lat = []
while True:
    ok, frame = cap.read()
    if not ok: break
    if n % args.stride: n += 1; continue
    t0 = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = pose.predict(frame, imgsz=GEOMETRY.pose_imgsz, conf=0.35, verbose=False, device=0)[0]
    people = []
    if r.keypoints is not None and len(r.boxes):
        kps = r.keypoints.data.cpu().numpy()
        people = build_rois(r.boxes.xyxy.cpu().numpy(), kps)
        fr_scores = [frontality(k) for k in kps]
    ims, slots = [], []
    for i,p in enumerate(people):
        if p.head is not None and not p.head_collision:
            c = crop_of(rgb, p.head, GEOMETRY.head_crop_px)
            if c is not None: ims.append(c); slots.append((i,"head"))
        for hi,hb in enumerate(p.hands[:2]):
            c = crop_of(rgb, hb, GEOMETRY.hand_crop_px)
            if c is not None: ims.append(c); slots.append((i,f"hand{hi}"))
    E = embed_pil(ims) if ims else np.zeros((0,768),np.float32)
    pred = collections.defaultdict(dict)
    for (i,kind), e in zip(slots, E):
        e = e[None]
        if kind == "head":
            pred[i]["head_cap"] = float(clf["head_cap"].predict_proba(e)[0,1])
            if fr_scores[i] >= args.frontality:
                pred[i]["mask"] = float(clf["mask"].predict_proba(e)[0,1])
        else:
            pred[i].setdefault("hands", []).append(float(clf["gloves"].predict_proba(e)[0,1]))
    lat.append((time.time()-t0)*1000)

    npersons.append(len(people))
    for i,p in enumerate(people):
        v = {}
        v["cap"]  = ("present" if pred[i].get("head_cap",0) >= .5 else "absent") if "head_cap" in pred[i] else "unclear"
        v["mask"] = ("present" if pred[i].get("mask",0) >= .5 else "absent") if "mask" in pred[i] else "unclear"
        hs = pred[i].get("hands", [])
        v["gloves"] = "unclear" if len(hs) < 2 else ("present" if all(h >= .5 for h in hs) else "absent")
        for k_,s_ in v.items(): stats[f"{k_}:{s_}"] += 1
        col = BAD if "absent" in v.values() else (OK if "present" in v.values() else UNK)
        x1,y1,x2,y2 = [int(t) for t in p.person_box]
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
        if p.head is not None:
            hx1,hy1,hx2,hy2 = [int(t) for t in p.head]
            cv2.rectangle(frame,(hx1,hy1),(hx2,hy2),col,1)
        lines = [f"{'OK' if v[k]=='present' else ('X' if v[k]=='absent' else '?')} {k}" for k in ("cap","mask","gloves")]
        ty = max(0, y1-42)
        cv2.rectangle(frame,(x1,ty),(x1+82,ty+40),(18,24,28),-1)
        for j,t in enumerate(lines):
            c2 = OK if t.startswith("OK") else (BAD if t.startswith("X") else UNK)
            cv2.putText(frame,t,(x1+3,ty+12+13*j),cv2.FONT_HERSHEY_SIMPLEX,0.36,c2,1)
    cv2.rectangle(frame,(0,0),(W,24),(18,24,28),-1)
    cv2.putText(frame,f"ours: pose -> ROI -> probe   frame {n}   {len(people)} persons",
                (8,17),cv2.FONT_HERSHEY_SIMPLEX,0.5,(235,240,242),1)
    wr.write(frame); n += 1
    if n % 200 == 0: print(f"  {n}/{total}", flush=True)
cap.release(); wr.release()

f = len(npersons)
print(f"\nprocessed {f} frames, {np.mean(lat):.0f} ms/frame ({1000/np.mean(lat):.1f} fps)")
print(f"persons/frame: mean {np.mean(npersons):.2f}, max {max(npersons)}")
print(f"\n{'attribute':<10}{'present':>9}{'absent':>9}{'unclear':>9}{'decided':>10}")
for k in ("cap","mask","gloves"):
    p_,a_,u_ = stats[f"{k}:present"], stats[f"{k}:absent"], stats[f"{k}:unclear"]
    print(f"{k:<10}{p_:>9}{a_:>9}{u_:>9}{100*(p_+a_)/max(p_+a_+u_,1):>9.0f}%")
print(f"\nwrote {args.out}")
