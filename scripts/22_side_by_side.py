"""Both models, same frames, same label design, side by side.

The two produce different things natively -- apd_mbg emits PPE boxes, ours emits
per-person verdicts -- so they cannot be compared as drawn. Here apd_mbg's
detections are resolved to per-person verdicts the way the deployed service does
it (assign_apd_smart: PPE centre inside a person box, positive class vs negative
class), and both panels then use one overlay design:

    person box coloured by verdict, and OK / X / ? per attribute
    red = any violation, green = all compliant, grey = nothing decidable
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
POS = {"cap":"head_cap","mask":"mask","gloves":"gloves"}
NEG = {"cap":"no_head_cap","mask":"no_mask","gloves":"no_gloves"}
OK, BAD, UNK = (80,200,120), (60,60,235), (150,155,160)
ATTRS = ("cap","mask","gloves")

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--out", default="artifacts/compare.mp4")
ap.add_argument("--stride", type=int, default=3); ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--max-frames", type=int, default=0)
args = ap.parse_args()
DEV, MID = "cuda", "facebook/dinov2-base"; CROPS = Path("artifacts/crops")

core = ov.Core(); ovdev = "GPU" if "GPU" in core.available_devices else "CPU"
mbg = core.compile_model(core.read_model("model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml"), ovdev)
mbg_out = mbg.outputs[0]
pose = YOLO(".work/yolo11x-pose.pt")
proc = AutoImageProcessor.from_pretrained(MID); enc = AutoModel.from_pretrained(MID).to(DEV).eval()

@torch.no_grad()
def embed(ims,b=64):
    o=[]
    for i in range(0,len(ims),b):
        o.append(enc(**proc(images=ims[i:i+b],return_tensors="pt").to(DEV)).last_hidden_state[:,0].float().cpu().numpy())
    return np.concatenate(o) if o else np.zeros((0,768),np.float32)

splits=json.loads(Path("artifacts/splits.json").read_text())["splits"]
fs={f:k for k,v in splits.items() for f in v}
idx=[json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]
hc={r["person_id"]:r["crop"] for r in idx if r["part"]=="head"}
hbx={(r["person_id"],int(r["crop"].rsplit("_h",1)[1].split(".")[0])):r["crop"] for r in idx if r["part"]=="hand"}
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
        c=hbx.get((r["person_id"],h["hand"]))
        if c and h["value"]!=Value.UNCLEAR.value:
            rows["gloves"][0].append(c); rows["gloves"][1].append(int(h["value"]=="present"))
for m in extra:
    if m.get("split")=="train" and m["value"]!="unclear" and m["person_id"] in hc:
        rows["mask"][0].append(hc[m["person_id"]]); rows["mask"][1].append(int(m["value"]=="present"))
clf={a:LogisticRegression(max_iter=3000,C=0.05,class_weight="balanced").fit(
        embed([Image.open(CROPS/p).convert("RGB") for p in ps]),np.array(ys)) for a,(ps,ys) in rows.items()}
print("probes fitted:", {a:f"{int(sum(y))}+/{len(y)-int(sum(y))}-" for a,(_,y) in rows.items()})

def letterbox(img,s=640):
    h,w=img.shape[:2]; r=min(s/h,s/w); nh,nw=round(h*r),round(w*r)
    o=np.full((s,s,3),114,np.uint8); t,l=(s-nh)//2,(s-nw)//2
    o[t:t+nh,l:l+nw]=cv2.resize(img,(nw,nh)); return o,r,l,t
def crop_of(img,box,size):
    x1,y1,x2,y2=box; pad=0.35*max(x2-x1,y2-y1)
    c=img[max(0,int(y1-pad)):int(y2+pad),max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else Image.fromarray(cv2.resize(c,(size,size)))
def centre_in(b,p):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
    return p[0]<=cx<=p[2] and p[1]<=cy<=p[3]

def draw(panel, people, title, sub):
    H_,W_ = panel.shape[:2]
    for pbox,v in people:
        col = BAD if "absent" in v.values() else (OK if "present" in v.values() else UNK)
        x1,y1,x2,y2=[int(t) for t in pbox]
        cv2.rectangle(panel,(x1,y1),(x2,y2),col,2)
        ty=max(26,y1-44)
        cv2.rectangle(panel,(x1,ty),(x1+80,ty+42),(18,24,28),-1)
        for j,k in enumerate(ATTRS):
            s=v[k]; mark="OK" if s=="present" else ("X" if s=="absent" else "?")
            c2=OK if s=="present" else (BAD if s=="absent" else UNK)
            cv2.putText(panel,f"{mark} {k}",(x1+4,ty+13+13*j),cv2.FONT_HERSHEY_SIMPLEX,0.36,c2,1)
    cv2.rectangle(panel,(0,0),(W_,24),(18,24,28),-1)
    cv2.putText(panel,title,(8,17),cv2.FONT_HERSHEY_SIMPLEX,0.52,(235,240,242),1)
    cv2.putText(panel,sub,(W_-260,17),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,200,220),1)
    return panel

cap=cv2.VideoCapture(args.video)
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS=cap.get(cv2.CAP_PROP_FPS) or 25; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
wr=cv2.VideoWriter(args.out,cv2.VideoWriter_fourcc(*"mp4v"),FPS/args.stride,(W*2,H))
print(f"video {W}x{H} -> {W*2}x{H}  {args.out}")

n=0; stat=collections.defaultdict(collections.Counter); pc={"a":[],"b":[]}
while True:
    ok,fr=cap.read()
    if not ok: break
    if n%args.stride: n+=1; continue
    rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)

    # ---------- production: detections -> per-person verdicts ----------
    img,r,dx,dy=letterbox(rgb)
    p=mbg([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[mbg_out][0]
    if p.shape[0]<p.shape[1]: p=p.T
    sc=p[:,4:11]; cl=sc.argmax(1); cf=sc[np.arange(len(sc)),cl]; k=cf>=args.conf
    b,cf,cl=p[k][:,:4],cf[k],cl[k]; boxes=np.zeros((0,4))
    if len(b):
        xy=np.stack([b[:,0]-b[:,2]/2,b[:,1]-b[:,3]/2,b[:,0]+b[:,2]/2,b[:,1]+b[:,3]/2],1)
        xy[:,[0,2]]-=dx; xy[:,[1,3]]-=dy; xy/=r
        ii=cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],cf.tolist(),args.conf,0.7)
        ii=np.array(ii).flatten() if len(ii) else np.zeros(0,int); boxes,cl=xy[ii],cl[ii]
    pboxes=[boxes[i] for i in range(len(cl)) if cl[i]==0]
    ppe=[(boxes[i],LABELS[cl[i]]) for i in range(len(cl)) if cl[i]!=0]
    A=[]
    for pb in pboxes:
        v={}
        for a in ATTRS:
            has_p=any(nm==POS[a] and centre_in(bb,pb) for bb,nm in ppe)
            has_n=any(nm==NEG[a] and centre_in(bb,pb) for bb,nm in ppe)
            v[a]="absent" if has_n else ("present" if has_p else "unclear")
            stat["a"][f"{a}:{v[a]}"]+=1
        A.append((pb,v))
    pc["a"].append(len(pboxes))

    # ---------- ours: pose -> ROI -> probe ----------
    rr=pose.predict(fr,imgsz=GEOMETRY.pose_imgsz,conf=0.35,verbose=False,device=0)[0]
    people=[]; frs=[]
    if rr.keypoints is not None and len(rr.boxes):
        kp=rr.keypoints.data.cpu().numpy(); people=build_rois(rr.boxes.xyxy.cpu().numpy(),kp); frs=[frontality(x) for x in kp]
    ims,slots=[],[]
    for i,pp in enumerate(people):
        if pp.head is not None and not pp.head_collision:
            c=crop_of(rgb,pp.head,GEOMETRY.head_crop_px)
            if c is not None: ims.append(c); slots.append((i,"head"))
        for hi,hh in enumerate(pp.hands[:2]):
            c=crop_of(rgb,hh,GEOMETRY.hand_crop_px)
            if c is not None: ims.append(c); slots.append((i,"hand"))
    E=embed(ims) if ims else np.zeros((0,768),np.float32)
    per=collections.defaultdict(dict)
    for (i,kind),e in zip(slots,E):
        e=e[None]
        if kind=="head":
            per[i]["cap"]=clf["head_cap"].predict_proba(e)[0,1]
            if frs[i]>=0.5: per[i]["mask"]=clf["mask"].predict_proba(e)[0,1]
        else: per[i].setdefault("h",[]).append(clf["gloves"].predict_proba(e)[0,1])
    B=[]
    for i,pp in enumerate(people):
        d=per[i]; v={}
        v["cap"]="unclear" if "cap" not in d else ("present" if d["cap"]>=.5 else "absent")
        v["mask"]="unclear" if "mask" not in d else ("present" if d["mask"]>=.5 else "absent")
        hs=d.get("h",[])
        v["gloves"]="unclear" if len(hs)<2 else ("present" if all(x>=.5 for x in hs) else "absent")
        for a in ATTRS: stat["b"][f"{a}:{v[a]}"]+=1
        B.append((pp.person_box,v))
    pc["b"].append(len(people))

    left=draw(fr.copy(),A,"apd_mbg  YOLO11s  detect -> assign",f"{len(A)} persons")
    right=draw(fr.copy(),B,"ours  pose -> ROI -> probe",f"{len(B)} persons")
    wr.write(np.hstack([left,right])); n+=1
    if n%150==0: print(f"  {n}/{total}",flush=True)
    if args.max_frames and n>=args.max_frames: break
cap.release(); wr.release()

f=len(pc["a"])
print(f"\n{f} frames, identical overlay\n")
print(f"{'':<22}{'apd_mbg':>12}{'ours':>10}")
print(f"{'persons / frame':<22}{np.mean(pc['a']):>12.2f}{np.mean(pc['b']):>10.2f}")
for a in ATTRS:
    print(f"\n  {a}")
    for s in ("present","absent","unclear"):
        print(f"    {s:<18}{stat['a'][f'{a}:{s}']:>12}{stat['b'][f'{a}:{s}']:>10}")
print(f"\nwrote {args.out}")
