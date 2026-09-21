"""Head-to-head on identical frames: apd_mbg (production) vs ours.

Same frames, same video. The two disagree about something more basic than PPE,
so person count is reported first.
"""
import argparse, collections, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np, torch, openvino as ov
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
from apd.pose import build_rois, frontality
from apd.schema import GEOMETRY, Value

LABELS = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--conf", type=float, default=0.30)
args = ap.parse_args()
DEV, MID = "cuda", "facebook/dinov2-base"; CROPS = Path("artifacts/crops")

core = ov.Core(); ovdev = "GPU" if "GPU" in core.available_devices else "CPU"
mbg = core.compile_model(core.read_model("model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml"), ovdev)
mbg_out = mbg.outputs[0]
pose = YOLO(".work/yolo11x-pose.pt")
proc = AutoImageProcessor.from_pretrained(MID); enc = AutoModel.from_pretrained(MID).to(DEV).eval()

@torch.no_grad()
def embed(ims, b=64):
    o=[]
    for i in range(0,len(ims),b):
        o.append(enc(**proc(images=ims[i:i+b],return_tensors="pt").to(DEV)).last_hidden_state[:,0].float().cpu().numpy())
    return np.concatenate(o) if o else np.zeros((0,768),np.float32)

splits=json.loads(Path("artifacts/splits.json").read_text())["splits"]
fs={f:k for k,v in splits.items() for f in v}
idx=[json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]
hc={r["person_id"]:r["crop"] for r in idx if r["part"]=="head"}
hb={(r["person_id"],int(r["crop"].rsplit("_h",1)[1].split(".")[0])):r["crop"] for r in idx if r["part"]=="hand"}
seed=[json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()]
extra=[m for m in (json.loads(l) for l in Path("artifacts/mined_labels.jsonl").read_text().splitlines()) if m["attribute"]=="mask"]
extra+=[b for b in (json.loads(l) for l in Path("artifacts/bulk_labels.jsonl").read_text().splitlines()) if b.get("attribute")=="mask"]
rows={a:([],[]) for a in ("head_cap","mask","gloves")}
for r in seed:
    if fs.get(r["frame"])!="train": continue
    for a in ("head_cap","mask"):
        v=r["labels"][a]
        if v!=Value.UNCLEAR.value and r["person_id"] in hc:
            rows[a][0].append(hc[r["person_id"]]); rows[a][1].append(int(v=="present"))
    for h in r.get("hands",[]):
        c=hb.get((r["person_id"],h["hand"]))
        if c and h["value"]!=Value.UNCLEAR.value:
            rows["gloves"][0].append(c); rows["gloves"][1].append(int(h["value"]=="present"))
for m in extra:
    if m.get("split")=="train" and m["value"]!="unclear" and m["person_id"] in hc:
        rows["mask"][0].append(hc[m["person_id"]]); rows["mask"][1].append(int(m["value"]=="present"))
clf={a:LogisticRegression(max_iter=3000,C=0.05,class_weight="balanced").fit(
        embed([Image.open(CROPS/p).convert("RGB") for p in ps]), np.array(ys))
     for a,(ps,ys) in rows.items()}

def letterbox(img,s=640):
    h,w=img.shape[:2]; r=min(s/h,s/w); nh,nw=round(h*r),round(w*r)
    o=np.full((s,s,3),114,np.uint8); t,l=(s-nh)//2,(s-nw)//2
    o[t:t+nh,l:l+nw]=cv2.resize(img,(nw,nh)); return o,r,l,t
def crop_of(img,box,size):
    x1,y1,x2,y2=box; pad=0.35*max(x2-x1,y2-y1)
    c=img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else Image.fromarray(cv2.resize(c,(size,size)))

cap=cv2.VideoCapture(args.video); n=0
A=collections.Counter(); B=collections.Counter(); pa=[]; pb=[]; ta=[]; tb=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    if n%args.stride: n+=1; continue
    rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
    # --- production
    t0=time.time(); img,r,dx,dy=letterbox(rgb)
    p=mbg([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[mbg_out][0]
    if p.shape[0]<p.shape[1]: p=p.T
    sc=p[:,4:11]; cl=sc.argmax(1); cf=sc[np.arange(len(sc)),cl]; k=cf>=args.conf
    b,cf,cl=p[k][:,:4],cf[k],cl[k]
    if len(b):
        xy=np.stack([b[:,0]-b[:,2]/2,b[:,1]-b[:,3]/2,b[:,0]+b[:,2]/2,b[:,1]+b[:,3]/2],1)
        ii=cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],cf.tolist(),args.conf,0.7)
        ii=np.array(ii).flatten() if len(ii) else np.zeros(0,int); cl=cl[ii]
    ta.append((time.time()-t0)*1000)
    for c in cl: A[LABELS[c]]+=1
    pa.append(int((cl==0).sum()) if len(cl) else 0)
    # --- ours
    t0=time.time()
    rr=pose.predict(fr,imgsz=GEOMETRY.pose_imgsz,conf=0.35,verbose=False,device=0)[0]
    people=[]; frs=[]
    if rr.keypoints is not None and len(rr.boxes):
        kp=rr.keypoints.data.cpu().numpy(); people=build_rois(rr.boxes.xyxy.cpu().numpy(),kp); frs=[frontality(k) for k in kp]
    ims,slots=[],[]
    for i,pp in enumerate(people):
        if pp.head is not None and not pp.head_collision:
            c=crop_of(rgb,pp.head,GEOMETRY.head_crop_px)
            if c is not None: ims.append(c); slots.append((i,"head"))
        for hi,hbx in enumerate(pp.hands[:2]):
            c=crop_of(rgb,hbx,GEOMETRY.hand_crop_px)
            if c is not None: ims.append(c); slots.append((i,"hand"))
    E=embed(ims) if ims else np.zeros((0,768),np.float32)
    per=collections.defaultdict(dict)
    for (i,kind),e in zip(slots,E):
        e=e[None]
        if kind=="head":
            per[i]["cap"]=clf["head_cap"].predict_proba(e)[0,1]
            if frs[i]>=0.5: per[i]["mask"]=clf["mask"].predict_proba(e)[0,1]
        else: per[i].setdefault("h",[]).append(clf["gloves"].predict_proba(e)[0,1])
    tb.append((time.time()-t0)*1000)
    for i,_ in enumerate(people):
        d=per[i]
        B["cap:"+("present" if d.get("cap",0)>=.5 else "absent")if "cap" in d else "cap:unclear"]+=1
        B["mask:"+("present" if d.get("mask",0)>=.5 else "absent")if "mask" in d else "mask:unclear"]+=1
        hs=d.get("h",[])
        B["gloves:"+("unclear" if len(hs)<2 else ("present" if all(x>=.5 for x in hs) else "absent"))]+=1
    pb.append(len(people)); n+=1
cap.release()

f=len(pa)
print(f"\n{f} identical frames (every {args.stride}th)\n")
print(f"{'':<26}{'apd_mbg (prod)':>18}{'ours (pose)':>16}")
print(f"{'persons / frame':<26}{np.mean(pa):>18.2f}{np.mean(pb):>16.2f}")
print(f"{'ms / frame':<26}{np.mean(ta):>18.0f}{np.mean(tb):>16.0f}")
print(f"\n{'compliance among decided':<26}{'apd_mbg':>18}{'ours':>16}")
for ours,pos,neg in (("cap","head_cap","no_head_cap"),("mask","mask","no_mask"),("gloves","gloves","no_gloves")):
    a=A[pos]/max(A[pos]+A[neg],1)*100
    bp,bn=B[f"{ours}:present"],B[f"{ours}:absent"]
    b_=bp/max(bp+bn,1)*100
    print(f"  {ours:<24}{a:>17.0f}%{b_:>15.0f}%")
print(f"\n{'raw detections':<26}{'apd_mbg':>18}")
for c in LABELS: print(f"  {c:<24}{A[c]:>18}")
