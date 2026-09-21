"""Render every model's own labels to its own directory, for visual review.

    artifacts/viz/<model>/<site_code>/<id>.jpg

Each model is drawn in its native representation, because that is what "its
labels" means -- apd_mbg emits PPE boxes, the pipeline models emit a verdict per
person. Forcing them into one form would hide exactly the difference worth
looking at. A header strip on every frame carries the model name and its counts
so frames can be compared side by side in a file browser.
"""
import argparse, collections, csv, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch, openvino as ov
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd.overlay import clean, detect
from apd.pose import build_rois, frontality, roi_usable
from apd.schema import GEOMETRY

NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
DET_COL = {"person":(240,190,60),"gloves":(90,220,120),"head_cap":(120,200,255),
           "mask":(200,160,255),"no_gloves":(60,60,240),"no_head_cap":(60,120,240),
           "no_mask":(80,60,220)}
OK,BAD,UNK = (80,200,120),(60,60,235),(150,155,160)
ATTRS = ("head_cap","mask","gloves")
MASK_MARGIN = 0.25   # calibrated in scripts/49; replaces the frontality gate

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--out", default="artifacts/viz")
ap.add_argument("--models", default="apd_mbg,voter3_ours,voter4_ours")
a = ap.parse_args()
want = set(a.models.split(","))

imgs = sorted(Path("data/sites").rglob("*.jpg"))
if a.limit: imgs = imgs[:a.limit]
meta = {r["ID Pelanggaran"]: r for r in csv.DictReader(open("artifacts/site_sample.csv"))}
print(f"{len(imgs)} images -> {a.out}/<model>/<site>/")

core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
mbg = core.compile_model(core.read_model("model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml"), dev)
port = mbg.outputs[0]
pose = YOLO(".work/yolo11x-pose.pt")

import pickle
probes = pickle.load(open("artifacts/voter3_ci3/probes.pkl","rb"))
from transformers import AutoImageProcessor, AutoModel
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
enc = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()
y26 = {k: YOLO(f"runs/classify/runs/classify/runs/ci3_{k}/weights/best.pt") for k in ATTRS}
y26_idx = {k: {v:i for i,v in m.names.items()} for k,m in y26.items()}

@torch.no_grad()
def embed(ims):
    if not ims: return np.zeros((0,768),np.float32)
    return enc(**proc(images=ims,return_tensors="pt").to("cuda")).last_hidden_state[:,0].float().cpu().numpy()

def letterbox(img,s=640):
    h,w=img.shape[:2]; r=min(s/h,s/w); nh,nw=round(h*r),round(w*r)
    o=np.full((s,s,3),114,np.uint8); t,l=(s-nh)//2,(s-nw)//2
    o[t:t+nh,l:l+nw]=cv2.resize(img,(nw,nh)); return o,r,l,t
def crop_np(img,box,size):
    x1,y1,x2,y2=box; pad=0.35*max(x2-x1,y2-y1)
    c=img[max(0,int(y1-pad)):int(y2+pad),max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

def header(im, text, sub=""):
    H,W=im.shape[:2]
    cv2.rectangle(im,(0,0),(W,22),(18,24,28),-1)
    cv2.putText(im,text,(6,15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(235,240,242),1)
    if sub:
        (tw,_),_=cv2.getTextSize(sub,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
        cv2.putText(im,sub,(max(6,W-tw-6),15),cv2.FONT_HERSHEY_SIMPLEX,0.42,(150,200,220),1)
    return im

def draw_verdicts(im, people):
    """Verdict cards placed so they do not cover each other.

    In a crowded frame the fixed above-the-box position stacked cards on top of
    one another and nothing was readable. Each card takes the first candidate
    slot that is in-frame and clear of the cards already placed, and a leader
    line keeps it attached to its person.
    """
    H, W = im.shape[:2]
    CW, CH = 86, 44
    placed = []

    def free(x, y):
        if x < 2 or y < 24 or x + CW > W - 2 or y + CH > H - 2:
            return False
        return all(not (x < px + CW and x + CW > px and y < py + CH and y + CH > py)
                   for px, py in placed)

    order = sorted(range(len(people)), key=lambda i: people[i][0][0])
    for i in order:
        box, v = people[i]
        x1, y1, x2, y2 = [int(t) for t in box]
        col = BAD if "absent" in v.values() else (OK if "present" in v.values() else UNK)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)

        for cx, cy in ((x1, y1 - CH - 4), (x2 + 4, y1), (x1 - CW - 4, y1),
                       (x1, y2 + 4), (x2 + 4, y2 - CH), (x1, y1 + 4)):
            if free(cx, cy):
                break
        else:
            cx, cy = min(max(2, x1), W - CW - 2), 24
            while not free(cx, cy) and cy + CH < H - 2:
                cy += CH + 3
        placed.append((cx, cy))

        cv2.line(im, (x1 + 2, y1 + 2), (cx + CW // 2, cy + CH // 2), col, 1)
        cv2.rectangle(im, (cx, cy), (cx + CW, cy + CH), (18, 24, 28), -1)
        cv2.rectangle(im, (cx, cy), (cx + CW, cy + CH), col, 1)
        for j, k in enumerate(ATTRS):
            st = v[k]; mk = {"present": "OK", "absent": "X", "unclear": "?"}[st]
            c2 = OK if st == "present" else (BAD if st == "absent" else UNK)
            cv2.putText(im, f"{mk} {k}", (cx + 4, cy + 13 + 13 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, c2, 1)
    return im


t0=time.time(); counts=collections.Counter()
for n,p in enumerate(imgs,1):
    bgr=cv2.imread(str(p))
    if bgr is None: continue
    # Draw on the cleaned frame: leaving the dashboard's own boxes and captions
    # in the picture makes it impossible to tell our labels from its.
    _ov,_ovb=detect(bgr)
    bgr=clean(bgr,(_ov,_ovb))
    site=p.parent.name; flag=meta.get(p.stem,{}).get("violations","")

    if "apd_mbg" in want:
        img,r,dx,dy=letterbox(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        o=mbg([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
        if o.shape[0]<o.shape[1]: o=o.T
        sc=o[:,4:11]; cl=sc.argmax(1); cf=sc.max(1); k=cf>=a.conf
        b,cf,cl=o[k][:,:4],cf[k],cl[k]; im=bgr.copy(); c=collections.Counter()
        if len(b):
            xy=np.stack([b[:,0]-b[:,2]/2,b[:,1]-b[:,3]/2,b[:,0]+b[:,2]/2,b[:,1]+b[:,3]/2],1)
            xy[:,[0,2]]-=dx; xy[:,[1,3]]-=dy; xy/=r
            idx=cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                                  cf.tolist(),a.conf,0.7)
            idx=np.array(idx).flatten() if len(idx) else np.zeros(0,int)
            for i in idx:
                nm=NAMES[cl[i]]; c[nm]+=1; col=DET_COL[nm]
                x1,y1,x2,y2=[int(t) for t in xy[i]]
                cv2.rectangle(im,(x1,y1),(x2,y2),col,2)
                tag=f"{nm} {cf[i]:.2f}"
                (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
                cv2.rectangle(im,(x1,max(22,y1-th-5)),(x1+tw+5,max(34,y1)),col,-1)
                cv2.putText(im,tag,(x1+3,max(32,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(20,20,20),1)
        header(im,f"apd_mbg  {site}",f"flag: {flag}  det: {sum(c.values())}")
        d=Path(a.out)/"apd_mbg"/site; d.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(d/p.name),im); counts["apd_mbg"]+=1

    if {"voter3_ours","voter4_ours"} & want:
        rr=pose.predict(bgr,imgsz=1280,conf=0.40,verbose=False,device=0)[0]
        people=[]; frs=[]
        if rr.keypoints is not None and len(rr.boxes):
            kp=rr.keypoints.data.cpu().numpy()
            people=build_rois(rr.boxes.xyxy.cpu().numpy(),kp); frs=[frontality(x) for x in kp]
        heads,hands=[],[]
        for i,pp in enumerate(people):
            _pb=[q.person_box for q in people]
            _ok = roi_usable(pp.head,bgr.shape,part="head",overlay=_ov,others=_pb,self_box=pp.person_box)
            hc = (crop_np(bgr,pp.head,GEOMETRY.head_crop_px)
                  if (_ok and not pp.head_collision) else None)
            heads.append(hc)
            hands.append([crop_np(bgr,h,GEOMETRY.hand_crop_px) for h in pp.hands[:2]
                          if roi_usable(h,bgr.shape,part="hand",overlay=_ov,
                                        others=_pb,self_box=pp.person_box)])

        if "voter3_ours" in want:
            ims=[Image.fromarray(cv2.cvtColor(h,cv2.COLOR_BGR2RGB)) for h in heads if h is not None]
            hidx=[i for i,h in enumerate(heads) if h is not None]
            E=embed(ims); res=collections.defaultdict(dict)
            for i,e in zip(hidx,E):
                res[i]["head_cap"]=probes["head_cap"].predict_proba(e[None])[0,1]
                res[i]["mask"]=probes["mask"].predict_proba(e[None])[0,1]
            hims=[]; hmap=[]
            for i,hl in enumerate(hands):
                for c_ in hl:
                    if c_ is not None: hims.append(Image.fromarray(cv2.cvtColor(c_,cv2.COLOR_BGR2RGB))); hmap.append(i)
            HE=embed(hims)
            for i,e in zip(hmap,HE):
                res[i].setdefault("h",[]).append(probes["gloves"].predict_proba(e[None])[0,1])
            out=[]
            for i,pp in enumerate(people):
                d_=res[i]; v={}
                v["head_cap"]="unclear" if "head_cap" not in d_ else ("present" if d_["head_cap"]>=.5 else "absent")
                v["mask"]="unclear" if "mask" not in d_ else ("present" if d_["mask"]>=.5 else "absent")
                hs=d_.get("h",[])
                v["gloves"]="unclear" if len(hs)<2 else ("present" if all(x>=.5 for x in hs) else "absent")
                out.append((pp.person_box,v))
            im=draw_verdicts(bgr.copy(),out)
            header(im,f"voter3-ours  pose+DINOv2  {site}",f"flag: {flag}  persons: {len(out)}")
            d=Path(a.out)/"voter3_ours"/site; d.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(d/p.name),im); counts["voter3_ours"]+=1

        if "voter4_ours" in want:
            out=[]
            for i,pp in enumerate(people):
                v={}
                hc=heads[i]
                for attr in ("head_cap","mask"):
                    if hc is None: v[attr]="unclear"; continue
                    r_=y26[attr].predict(hc,verbose=False,device=0)[0]
                    pr=float(r_.probs.data[y26_idx[attr]["present"]])
                    if attr=="mask" and abs(pr-.5)<MASK_MARGIN: v[attr]="unclear"; continue
                    v[attr]="present" if pr>=.5 else "absent"
                hp=[c_ for c_ in hands[i] if c_ is not None]
                if len(hp)<2: v["gloves"]="unclear"
                else:
                    gs=[float(y26["gloves"].predict(c_,verbose=False,device=0)[0].probs.data[y26_idx["gloves"]["present"]]) for c_ in hp]
                    v["gloves"]="present" if all(g>=.5 for g in gs) else "absent"
                out.append((pp.person_box,v))
            im=draw_verdicts(bgr.copy(),out)
            header(im,f"voter4-ours  pose+YOLO26  {site}",f"flag: {flag}  persons: {len(out)}")
            d=Path(a.out)/"voter4_ours"/site; d.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(d/p.name),im); counts["voter4_ours"]+=1

    if n%150==0: print(f"  {n}/{len(imgs)}  {n/(time.time()-t0):.1f} img/s",flush=True)

print(f"\n{time.time()-t0:.0f}s   {dict(counts)}")
for m in sorted(want): print(f"  {a.out}/{m}/")
