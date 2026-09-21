"""Evaluate our pipeline on the 53 annotated frames, honestly split.

Two parts, because the two stages have different contamination status.

  person detection -- our stage 1 is off-the-shelf yolo11x-pose, never fine-tuned
      on any of this, so all 53 frames are fair game and directly comparable with
      the production detector.

  PPE attributes -- the probes were fitted on labels derived from these frames,
      so only the 14 frames outside our train split can be used. Small, and
      reported as such rather than quietly averaged over all 53.

Attribute ground truth is derived the same way for both models: a GT PPE box
whose centre falls inside a GT person box makes that attribute present.
"""
import argparse, collections, contextlib, io, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np, torch, openvino as ov
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
from apd.pose import build_rois, frontality
from apd.schema import GEOMETRY, Value

GT_CAT={"person":3,"gloves":4,"head_cap":5,"mask":6}
ATTR=("head_cap","mask","gloves")
ap=argparse.ArgumentParser(); ap.add_argument("--op",type=float,default=0.30); a=ap.parse_args()
ROOT=Path("data/sppg"); DEV,MID="cuda","facebook/dinov2-base"; CROPS=Path("artifacts/crops")

raw=json.loads((ROOT/".exports/coco-1778210446.4251382.json").read_text())
annimgs={x["image_id"] for x in raw["annotations"]}
sub={"images":[i for i in raw["images"] if i["id"] in annimgs],
     "annotations":[dict(x,iscrowd=0) for x in raw["annotations"]],"categories":raw["categories"]}
Path("artifacts/_gt_53.json").write_text(json.dumps(sub))
with contextlib.redirect_stdout(io.StringIO()): coco=COCO("artifacts/_gt_53.json")
imgs={i["id"]:i for i in sub["images"]}
sp=json.loads(Path("artifacts/splits.json").read_text())["splits"]
fs={f:k for k,v in sp.items() for f in v}
held={i for i,v in imgs.items() if fs.get(v["file_name"])!="train"}
print(f"53 annotated frames; {len(held)} held out from our training\n")

pose=YOLO(".work/yolo11x-pose.pt")
proc=AutoImageProcessor.from_pretrained(MID); enc=AutoModel.from_pretrained(MID).to(DEV).eval()
@torch.no_grad()
def embed(ims,b=64):
    o=[]
    for i in range(0,len(ims),b):
        o.append(enc(**proc(images=ims[i:i+b],return_tensors="pt").to(DEV)).last_hidden_state[:,0].float().cpu().numpy())
    return np.concatenate(o) if o else np.zeros((0,768),np.float32)

idx=[json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]
hcp={r["person_id"]:r["crop"] for r in idx if r["part"]=="head"}
hnd={(r["person_id"],int(r["crop"].rsplit("_h",1)[1].split(".")[0])):r["crop"] for r in idx if r["part"]=="hand"}
rows={k:([],[]) for k in ATTR}
for r in (json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()):
    if fs.get(r["frame"])!="train": continue
    for k in ("head_cap","mask"):
        v=r["labels"][k]
        if v!=Value.UNCLEAR.value and r["person_id"] in hcp:
            rows[k][0].append(hcp[r["person_id"]]); rows[k][1].append(int(v=="present"))
    for h in r.get("hands",[]):
        c=hnd.get((r["person_id"],h["hand"]))
        if c and h["value"]!=Value.UNCLEAR.value:
            rows["gloves"][0].append(c); rows["gloves"][1].append(int(h["value"]=="present"))
for m in (json.loads(l) for l in Path("artifacts/mined_labels.jsonl").read_text().splitlines()):
    if m["attribute"]=="mask" and m.get("split")=="train" and m["value"]!="unclear" and m["person_id"] in hcp:
        rows["mask"][0].append(hcp[m["person_id"]]); rows["mask"][1].append(int(m["value"]=="present"))
clf={k:LogisticRegression(max_iter=3000,C=0.05,class_weight="balanced").fit(
        embed([Image.open(CROPS/p).convert("RGB") for p in ps]),np.array(ys)) for k,(ps,ys) in rows.items()}
print("probes:",{k:f"{int(sum(y))}+/{len(y)-int(sum(y))}-" for k,(_,y) in rows.items()},"\n")

def crop_of(img,box,size):
    x1,y1,x2,y2=box; pad=0.35*max(x2-x1,y2-y1)
    c=img[max(0,int(y1-pad)):int(y2+pad),max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else Image.fromarray(cv2.resize(c,(size,size)))
def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ix,iy=max(ax,bx),max(ay,by); jx,jy=min(ax+aw,bx+bw),min(ay+ah,by+bh)
    inter=max(0,jx-ix)*max(0,jy-iy); u=aw*ah+bw*bh-inter
    return inter/u if u>0 else 0.

pdets=[]; verdicts={}
for iid,im in imgs.items():
    bgr=cv2.imread(str(ROOT/im["file_name"])); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    r=pose.predict(bgr,imgsz=GEOMETRY.pose_imgsz,conf=0.15,verbose=False,device=0)[0]
    if r.keypoints is None or not len(r.boxes): continue
    kp=r.keypoints.data.cpu().numpy(); bx=r.boxes.xyxy.cpu().numpy(); cf=r.boxes.conf.cpu().numpy()
    people=build_rois(bx,kp); frs=[frontality(k) for k in kp]
    for b,c in zip(bx,cf):
        pdets.append({"image_id":int(iid),"category_id":GT_CAT["person"],
                      "bbox":[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])],"score":float(c)})
    ims,slots=[],[]
    for i,p in enumerate(people):
        if p.head is not None and not p.head_collision:
            cr=crop_of(rgb,p.head,GEOMETRY.head_crop_px)
            if cr is not None: ims.append(cr); slots.append((i,"head"))
        for hi,hb in enumerate(p.hands[:2]):
            cr=crop_of(rgb,hb,GEOMETRY.hand_crop_px)
            if cr is not None: ims.append(cr); slots.append((i,"hand"))
    E=embed(ims) if ims else np.zeros((0,768),np.float32)
    per=collections.defaultdict(dict)
    for (i,kind),e in zip(slots,E):
        e=e[None]
        if kind=="head":
            per[i]["head_cap"]=clf["head_cap"].predict_proba(e)[0,1]
            if frs[i]>=0.5: per[i]["mask"]=clf["mask"].predict_proba(e)[0,1]
        else: per[i].setdefault("h",[]).append(clf["gloves"].predict_proba(e)[0,1])
    out=[]
    for i,p in enumerate(people):
        d=per[i]; v={}
        v["head_cap"]="unclear" if "head_cap" not in d else ("present" if d["head_cap"]>=.5 else "absent")
        v["mask"]="unclear" if "mask" not in d else ("present" if d["mask"]>=.5 else "absent")
        hs=d.get("h",[])
        v["gloves"]="unclear" if len(hs)<2 else ("present" if all(x>=.5 for x in hs) else "absent")
        out.append((p.person_box,v))
    verdicts[iid]=out

# ---------- 1. person detection, all 53 frames ----------
with contextlib.redirect_stdout(io.StringIO()):
    e=COCOeval(coco,coco.loadRes(pdets),"bbox"); e.params.imgIds=list(imgs); e.evaluate(); e.accumulate()
ci=list(coco.getCatIds()).index(GT_CAT["person"])
pr=e.eval["precision"][0,:,ci,0,2]; ap50=float(np.mean(pr[pr>-1]))
pa=e.eval["precision"][:,:,ci,0,2]; apa=float(np.mean(pa[pa>-1]))
d_op=[d for d in pdets if d["score"]>=a.op]; tp=0; used=collections.defaultdict(set)
gts={i:[coco.anns[x]["bbox"] for x in coco.getAnnIds(imgIds=[i],catIds=[GT_CAT["person"]])] for i in imgs}
for d in sorted(d_op,key=lambda x:-x["score"]):
    best,bi=0,None
    for j,g in enumerate(gts[d["image_id"]]):
        if j in used[d["image_id"]]: continue
        v=iou(d["bbox"],g)
        if v>best: best,bi=v,j
    if best>=0.5: tp+=1; used[d["image_id"]].add(bi)
ngt=sum(len(v) for v in gts.values())
print("PERSON DETECTION -- all 53 frames (our stage 1 is off-the-shelf, uncontaminated)")
print(f"  {'model':<22}{'GT':>5}{'AP@.5':>8}{'AP@.5:.95':>11}{'P@op':>7}{'R@op':>7}")
print(f"  {'ours (yolo11x-pose)':<22}{ngt:>5}{ap50:>8.3f}{apa:>11.3f}"
      f"{tp/len(d_op):>7.2f}{tp/ngt:>7.2f}")
print(f"  {'apd_mbg (production)':<22}{262:>5}{0.470:>8.3f}{0.168:>11.3f}{0.79:>7.2f}{0.35:>7.2f}")

# ---------- 2. attributes on held-out frames ----------
print(f"\nPPE ATTRIBUTES -- {len(held)} held-out frames only")
print(f"  {'attribute':<12}{'matched':>9}{'decided':>9}{'agree':>7}{'rate':>7}")
for k in ATTR:
    ok=tot=dec=0
    for iid in held:
        gp=[coco.anns[x]["bbox"] for x in coco.getAnnIds(imgIds=[iid],catIds=[GT_CAT["person"]])]
        gk=[coco.anns[x]["bbox"] for x in coco.getAnnIds(imgIds=[iid],catIds=[GT_CAT[k]])]
        for gb in gp:
            truth = "present" if any(gb[0]<=q[0]+q[2]/2<=gb[0]+gb[2] and gb[1]<=q[1]+q[3]/2<=gb[1]+gb[3] for q in gk) else "absent"
            best,bv=0,None
            for pb,v in verdicts.get(iid,[]):
                bb=[pb[0],pb[1],pb[2]-pb[0],pb[3]-pb[1]]
                s=iou(bb,gb)
                if s>best: best,bv=s,v
            if best<0.5 or bv is None: continue
            tot+=1
            if bv[k]!="unclear":
                dec+=1; ok += (bv[k]==truth)
    print(f"  {k:<12}{tot:>9}{dec:>9}{ok:>7}{(100*ok/dec if dec else float('nan')):>6.0f}%")
print("\nAttribute truth = a GT PPE box centred inside the GT person box.")
