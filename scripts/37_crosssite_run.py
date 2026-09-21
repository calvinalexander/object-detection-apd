"""Cross-site run: how does the production model behave on 60 unseen sites?

There is no box ground truth here -- these are dashboard violation records, which
carry only a per-image violation type. So two proxies:

  domain health   person count from apd_mbg against the same count from the pose
      model. On its home domain the two agree closely; on the SPPG frames apd_mbg
      found 3.83/frame against pose's 8.35. The ratio is a cheap indicator of
      whether a site looks like training data or not.

  flag agreement  the CSV records which violation the dashboard raised. If
      apd_mbg detects that same negative class in that image, they agree. Weak
      evidence -- those flags likely came from this model family in the first
      place -- but it is what exists.
"""
import argparse, collections, csv, json, sys, time
from pathlib import Path
import cv2, numpy as np, openvino as ov
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO

NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
FLAG = {"head_cap":"no_head_cap", "mask":"no_mask", "gloves":"no_gloves"}
ap = argparse.ArgumentParser()
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="artifacts/crosssite.jsonl")
a = ap.parse_args()

rows = {r["ID Pelanggaran"]: r for r in csv.DictReader(open("artifacts/site_sample.csv"))}
imgs = sorted(Path("data/sites").rglob("*.jpg"))
if a.limit: imgs = imgs[:a.limit]
print(f"{len(imgs)} images across {len({p.parent.name for p in imgs})} sites")

core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
mbg = core.compile_model(core.read_model("model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml"), dev)
port = mbg.outputs[0]
pose = YOLO(".work/yolo11n-pose.pt")

def letterbox(img, s=640):
    h,w = img.shape[:2]; r = min(s/h, s/w); nh,nw = round(h*r), round(w*r)
    o = np.full((s,s,3),114,np.uint8); t,l = (s-nh)//2,(s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img,(nw,nh)); return o,r,l,t

out = open(a.out, "w"); t0 = time.time()
for n, p in enumerate(imgs, 1):
    bgr = cv2.imread(str(p))
    if bgr is None: continue
    img,r,dx,dy = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    o = mbg([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
    if o.shape[0] < o.shape[1]: o = o.T
    sc = o[:,4:11]; cl = sc.argmax(1); cf = sc.max(1)
    k = cf >= a.conf; b,cf,cl = o[k][:,:4], cf[k], cl[k]
    counts = collections.Counter()
    if len(b):
        xy = np.stack([b[:,0]-b[:,2]/2,b[:,1]-b[:,3]/2,b[:,0]+b[:,2]/2,b[:,1]+b[:,3]/2],1)
        idx = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                                cf.tolist(), a.conf, 0.7)
        idx = np.array(idx).flatten() if len(idx) else np.zeros(0,int)
        for i in idx: counts[NAMES[cl[i]]] += 1
    rp = pose.predict(bgr, imgsz=640, conf=0.30, verbose=False, device=0)[0]
    npose = int(len(rp.boxes)) if rp.boxes is not None else 0
    meta = rows.get(p.stem, {})
    out.write(json.dumps({"id":p.stem, "site":p.parent.name, "prov":meta.get("prov",""),
        "flagged": meta.get("violations",""), "mbg": dict(counts), "pose_persons": npose}) + "\n")
    if n % 200 == 0: print(f"  {n}/{len(imgs)}  {n/(time.time()-t0):.0f} img/s", flush=True)
out.close()
print(f"\n{len(imgs)} images in {time.time()-t0:.0f}s -> {a.out}")
