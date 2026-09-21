"""Three-model consensus on a shared person anchor.

apd_mbg misses the person entirely in 37% of these frames, which would make it
useless as a voter if its person boxes were the anchor. They are not: pose finds
the people for all three voters, and apd_mbg contributes only PPE class evidence,
assigned to those anchors the way the deployed service does it (centre-inside).
Its person-detection weakness then costs nothing, and its PPE head -- which
scores AP .87-.97 on its own domain -- still gets a vote.

Geometry and verdict come from different places, because the models are good at
different things. Measured against the production annotation: apd_mbg's PPE boxes
reach IoU .71-.82 and clear .5 on 90-96% of cases, while our anatomical ROIs reach
only .09-.21 -- they are generous crops sized by shoulder width, never intended to
localise PPE. So the box is apd_mbg's when it has one, and the verdict is the
ensemble's. Taking the smallest candidate was tried and is equivalent for head_cap
and mask but worse for gloves (.684 vs .713), since a hand ROI is sometimes
tighter than the glove itself.

Per person, per attribute, each voter returns present / absent / unclear.
Abstentions are not counted as votes; they reduce the denominator. A verdict is
auto-accepted only on unanimity among those that committed, with at least two
committing. Everything else goes to a review queue, which is the point --
disagreement is the signal for where a human is actually needed.
"""
import argparse, collections, csv, json, pickle, sys, time
from pathlib import Path
import cv2, numpy as np, torch, openvino as ov
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from apd.overlay import clean, detect
from apd.pose import build_rois, frontality, roi_usable
from apd.schema import GEOMETRY

NAMES=["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
POS={"head_cap":"head_cap","mask":"mask","gloves":"gloves"}
NEG={"head_cap":"no_head_cap","mask":"no_mask","gloves":"no_gloves"}
ATTRS=("head_cap","mask","gloves")
MASK_MARGIN=0.25   # calibrated in scripts/49 on the production val

ap=argparse.ArgumentParser()
ap.add_argument("--limit",type=int,default=0)
ap.add_argument("--conf",type=float,default=0.30)
ap.add_argument("--out",default="artifacts/consensus.jsonl")
a=ap.parse_args()

imgs=sorted(Path("data/sites").rglob("*.jpg"))
if a.limit: imgs=imgs[:a.limit]
meta={r["ID Pelanggaran"]:r for r in csv.DictReader(open("artifacts/site_sample.csv"))}

core=ov.Core(); dev="GPU" if "GPU" in core.available_devices else "CPU"
mbg=core.compile_model(core.read_model("model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml"),dev)
port=mbg.outputs[0]
pose=YOLO(".work/yolo11x-pose.pt")
probes=pickle.load(open("artifacts/voter3_ci3/probes.pkl","rb"))
proc=AutoImageProcessor.from_pretrained("facebook/dinov2-base")
enc=AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()
y26={k:YOLO(f"runs/classify/runs/classify/runs/ci3_{k}/weights/best.pt") for k in ATTRS}
y26i={k:{v:i for i,v in m.names.items()} for k,m in y26.items()}

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
def centre_in(b,p):
    cx,cy=b[0]+b[2]/2,b[1]+b[3]/2
    return p[0]<=cx<=p[2] and p[1]<=cy<=p[3]

out=open(a.out,"w"); t0=time.time()
tally=collections.Counter(); pair=collections.Counter(); skipped=collections.Counter()
for n,p in enumerate(imgs,1):
    bgr=cv2.imread(str(p))
    if bgr is None: continue
    # The dashboard's burned-in verdict is erased before any model sees the
    # frame: it names the answer and correlates with violation by construction.
    ov,_ovb=detect(bgr)
    bgr=clean(bgr,(ov,_ovb))
    # --- anchors: pose finds the people for everyone
    rr=pose.predict(bgr,imgsz=1280,conf=0.40,verbose=False,device=0)[0]
    if rr.keypoints is None or not len(rr.boxes): continue
    kp=rr.keypoints.data.cpu().numpy()
    people=build_rois(rr.boxes.xyxy.cpu().numpy(),kp); frs=[frontality(k) for k in kp]

    # --- voter A: apd_mbg PPE boxes assigned to those anchors
    img,r,dx,dy=letterbox(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    o=mbg([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
    if o.shape[0]<o.shape[1]: o=o.T
    sc=o[:,4:11]; cl=sc.argmax(1); cf=sc.max(1); k=cf>=a.conf
    b,cf,cl=o[k][:,:4],cf[k],cl[k]; ppe=[]
    if len(b):
        xy=np.stack([b[:,0]-b[:,2]/2,b[:,1]-b[:,3]/2,b[:,0]+b[:,2]/2,b[:,1]+b[:,3]/2],1)
        xy[:,[0,2]]-=dx; xy[:,[1,3]]-=dy; xy/=r
        idx=cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                              cf.tolist(),a.conf,0.7)
        idx=np.array(idx).flatten() if len(idx) else np.zeros(0,int)
        ppe=[(xy[i],NAMES[cl[i]]) for i in idx if NAMES[cl[i]]!="person"]

    # An ROI that cannot be seen is not classified. Previously only
    # head_collision was checked, so heads clipped by the frame edge, hidden
    # behind a caption, occluded by a nearer worker, or a couple of dozen
    # pixels across were all still answered -- confidently and often wrongly.
    pboxes=[pp.person_box for pp in people]
    heads=[]
    for pp in people:
        ck=roi_usable(pp.head,bgr.shape,part="head",overlay=ov,others=pboxes,self_box=pp.person_box)
        if not ck or pp.head_collision:
            heads.append(None); skipped[ck.reason if not ck else "head collision"]+=1
        else:
            heads.append(crop_np(bgr,pp.head,GEOMETRY.head_crop_px))
    hands=[]
    for pp in people:
        hs=[]
        for h in pp.hands[:2]:
            ck=roi_usable(h,bgr.shape,part="hand",overlay=ov,others=pboxes,self_box=pp.person_box)
            if not ck:
                skipped[ck.reason]+=1; continue
            hs.append(crop_np(bgr,h,GEOMETRY.hand_crop_px))
        hands.append(hs)

    # --- voter B: DINOv2 probes
    hidx=[i for i,h in enumerate(heads) if h is not None]
    E=embed([Image.fromarray(cv2.cvtColor(heads[i],cv2.COLOR_BGR2RGB)) for i in hidx])
    b3=collections.defaultdict(dict)
    for i,e in zip(hidx,E):
        b3[i]["head_cap"]=float(probes["head_cap"].predict_proba(e[None])[0,1])
        b3[i]["mask"]=float(probes["mask"].predict_proba(e[None])[0,1])
    him=[];hmap=[]
    for i,hl in enumerate(hands):
        for c_ in hl:
            if c_ is not None: him.append(Image.fromarray(cv2.cvtColor(c_,cv2.COLOR_BGR2RGB))); hmap.append(i)
    for i,e in zip(hmap,embed(him)):
        b3[i].setdefault("h",[]).append(float(probes["gloves"].predict_proba(e[None])[0,1]))

    recs=[]
    for i,pp in enumerate(people):
        pb=pp.person_box
        votes={}
        for at in ATTRS:
            v={}
            # apd_mbg
            hasp=any(nm==POS[at] and centre_in([bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]],pb) for bb,nm in ppe)
            hasn=any(nm==NEG[at] and centre_in([bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]],pb) for bb,nm in ppe)
            v["apd_mbg"]="unclear" if hasp==hasn else ("present" if hasp else "absent")
            # voter3
            if at=="gloves":
                hs=b3[i].get("h",[])
                v["voter3"]="unclear" if len(hs)<2 else ("present" if all(x>=.5 for x in hs) else "absent")
            else:
                # Mask abstains on confidence, not on head pose. The frontality
                # gate blocked 79% of instances; measured on the production val
                # it was discarding cases the voters get right 91% of the time,
                # 22 present and 31 absent, so it was withholding signal rather
                # than withholding guesses. A margin abstains on the cases that
                # are actually ambiguous to the model. (scripts/49)
                if at not in b3[i]:
                    v["voter3"]="unclear"
                else:
                    pr3=b3[i][at]
                    v["voter3"]=("unclear" if at=="mask" and abs(pr3-.5)<MASK_MARGIN
                                 else ("present" if pr3>=.5 else "absent"))
            # voter4
            if at=="gloves":
                hp=[c_ for c_ in hands[i] if c_ is not None]
                if len(hp)<2: v["voter4"]="unclear"
                else:
                    gs=[float(y26["gloves"].predict(c_,verbose=False,device=0)[0].probs.data[y26i["gloves"]["present"]]) for c_ in hp]
                    v["voter4"]="present" if all(g>=.5 for g in gs) else "absent"
            else:
                hc=heads[i]
                if hc is None: v["voter4"]="unclear"
                else:
                    pr=float(y26[at].predict(hc,verbose=False,device=0)[0].probs.data[y26i[at]["present"]])
                    v["voter4"]=("unclear" if at=="mask" and abs(pr-.5)<MASK_MARGIN
                                 else ("present" if pr>=.5 else "absent"))
            committed=[x for x in v.values() if x!="unclear"]
            if len(committed)>=2 and len(set(committed))==1:
                dec, kind = committed[0], f"unanimous_{len(committed)}"
            elif len(committed)>=2:
                c=collections.Counter(committed); top,cnt=c.most_common(1)[0]
                dec, kind = top, "majority" if cnt>len(committed)/2 else "tie"
            else:
                dec, kind = "unclear", "insufficient"
            tally[f"{at}:{kind}"]+=1
            for m1 in ("apd_mbg","voter3","voter4"):
                for m2 in ("apd_mbg","voter3","voter4"):
                    if m1<m2 and v[m1]!="unclear" and v[m2]!="unclear":
                        pair[f"{m1}~{m2}"]+= (v[m1]==v[m2]); pair[f"{m1}~{m2}#"]+=1
            # Geometry: apd_mbg's box for whichever class the vote settled on.
            box=None; src=None
            if dec in ("present","absent"):
                target = POS[at] if dec=="present" else NEG[at]
                cands=[bb for bb,nm in ppe
                       if nm==target and centre_in([bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]],pb)]
                if cands:
                    box=[round(float(x),1) for x in min(cands,key=lambda c_:(c_[2]-c_[0])*(c_[3]-c_[1]))]
                    src="apd_mbg"
                else:
                    roi = pp.head if at!="gloves" else (pp.hands[0] if pp.hands else None)
                    if roi is not None:
                        box=[round(float(x),1) for x in roi]; src="roi_fallback"
                tally[f"{at}:box_{src or 'none'}"]+=1
            votes[at]={"votes":v,"decision":dec,"kind":kind,"box":box,"box_src":src}
        recs.append({"person":i,"box":[round(float(x),1) for x in pb],"attrs":votes})
    out.write(json.dumps({"id":p.stem,"site":p.parent.name,
        "flagged":meta.get(p.stem,{}).get("violations",""),"persons":recs})+"\n")
    if n%150==0: print(f"  {n}/{len(imgs)}  {n/(time.time()-t0):.1f} img/s",flush=True)
out.close()

print(f"\n{len(imgs)} images in {time.time()-t0:.0f}s -> {a.out}\n")
print(f"{'attribute':<12}{'unanimous3':>12}{'unanimous2':>12}{'majority':>10}{'tie':>7}{'insufficient':>14}")
for at in ATTRS:
    # only the vote outcomes, not the box-provenance counters that share the prefix
    tot=sum(tally[f"{at}:{k}"] for k in
            ("unanimous_3","unanimous_2","majority","tie","insufficient"))
    row=[tally[f"{at}:unanimous_3"],tally[f"{at}:unanimous_2"],tally[f"{at}:majority"],
         tally[f"{at}:tie"],tally[f"{at}:insufficient"]]
    print(f"{at:<12}"+"".join(f"{x:>12}" if i<2 else (f"{x:>10}" if i==2 else (f"{x:>7}" if i==3 else f"{x:>14}"))
                              for i,x in enumerate(row))+f"   (n={tot})")
print(f"\nbox provenance for decided verdicts")
print(f"  {'attribute':<12}{'apd_mbg box':>13}{'roi fallback':>14}{'no geometry':>13}")
for at in ATTRS:
    print(f"  {at:<12}{tally[f'{at}:box_apd_mbg']:>13}{tally[f'{at}:box_roi_fallback']:>14}"
          f"{tally[f'{at}:box_none']:>13}")

print(f"\npairwise agreement where both committed")
for k in ("apd_mbg~voter3","apd_mbg~voter4","voter3~voter4"):
    d=pair[k+"#"]
    if d: print(f"  {k:<20}{pair[k]}/{d} = {100*pair[k]/d:.0f}%")
