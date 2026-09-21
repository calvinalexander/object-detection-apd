"""Is the frontality gate throwing away decidable mask cases?

Both voters abstain on mask unless the geometric frontality score clears 0.5,
which blocks 79% of person-instances. A face turned away genuinely cannot be
judged -- but 0.5 was chosen before there was anything to calibrate it against.
The production val has independent labels, so the trade the threshold makes
between coverage and correctness can be measured instead of assumed.
"""
import collections, json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from apd.pose import build_rois, frontality
from apd.schema import GEOMETRY, assert_contract

assert_contract()
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
root = Path("data/dump/dataset/apd_v3/dataset")
imgs = sorted(p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))

pose = YOLO(".work/yolo11x-pose.pt")
probes = pickle.load(open("artifacts/voter3_ci3/probes.pkl","rb"))
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
enc = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()
y26 = YOLO("runs/classify/runs/classify/runs/ci3_mask/weights/best.pt")
y26i = {v: i for i, v in y26.names.items()}

@torch.no_grad()
def embed(ims):
    if not ims: return np.zeros((0,768), np.float32)
    return enc(**proc(images=ims, return_tensors="pt").to("cuda")).last_hidden_state[:,0].float().cpu().numpy()

def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

def centre_in(b, p):
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

rows = []
for n, p in enumerate(imgs, 1):
    bgr = cv2.imread(str(p))
    if bgr is None: continue
    H, W = bgr.shape[:2]
    gtb = []
    lp = root/"labels/val"/(p.stem+".txt")
    if lp.exists():
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) < 5: continue
            c = int(f[0]); cx,cy,bw,bh = (float(v) for v in f[1:5])
            gtb.append(([(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H], NAMES[c]))
    rr = pose.predict(bgr, imgsz=1280, conf=0.40, verbose=False, device=0)[0]
    if rr.keypoints is None or not len(rr.boxes): continue
    kp = rr.keypoints.data.cpu().numpy()
    people = build_rois(rr.boxes.xyxy.cpu().numpy(), kp)
    for i, pp in enumerate(people):
        if pp.head is None or pp.head_collision: continue
        hc = crop_np(bgr, pp.head, GEOMETRY.head_crop_px)
        if hc is None: continue
        hasp = any(nm=="mask" and centre_in(bb, pp.person_box) for bb,nm in gtb)
        hasn = any(nm=="no_mask" and centre_in(bb, pp.person_box) for bb,nm in gtb)
        if hasp == hasn: continue
        e = embed([Image.fromarray(cv2.cvtColor(hc, cv2.COLOR_BGR2RGB))])[0]
        rows.append({
            "front": float(frontality(kp[i])),
            "true": "present" if hasp else "absent",
            "v3": float(probes["mask"].predict_proba(e[None])[0,1]),
            "v4": float(y26.predict(hc, verbose=False, device=0)[0].probs.data[y26i["present"]]),
        })
    if n % 40 == 0: print(f"  {n}/{len(imgs)}", flush=True)

print(f"\n{len(rows)} person-instances with a mask label and a usable head crop")
print(f"{'gate':>6} {'covered':>16} {'both agree':>12} {'agree correct':>14}")
for g in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    sub = [r for r in rows if r["front"] >= g]
    agree = [r for r in sub if (r["v3"] >= .5) == (r["v4"] >= .5)]
    ok = sum(1 for r in agree if ("present" if r["v3"] >= .5 else "absent") == r["true"])
    print(f"{g:6.1f} {len(sub):6d} ({len(sub)/max(1,len(rows)):5.1%}) "
          f"{len(agree):8d} ({len(agree)/max(1,len(sub)):5.1%}) "
          f"{ok:6d} ({ok/max(1,len(agree)):6.1%})")
Path("artifacts/mask_gate_sweep.json").write_text(json.dumps(rows))
