"""Our pose-anchored pipeline on the production val split -- the first common yardstick.

Two halves, with different contamination status:

  person detection -- yolo11x-pose off the shelf, never fine-tuned on anything
      here, so this is directly comparable with apd_mbg's number.

  attributes -- our probes were fitted on SPPG crops, a different camera domain
      entirely. This is therefore a cross-domain test of them, and should be read
      as such rather than as a like-for-like contest with a model trained on this
      data.

Attribute truth comes from the negative classes, which this split has and ours
never did: a `head_cap` box inside a person means present, a `no_head_cap` box
means absent, neither means unknown and is skipped rather than assumed.
"""
import collections, contextlib, io, json, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.pose import build_rois, frontality
from apd.schema import GEOMETRY, Value

NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
POS = {"head_cap":2, "mask":3, "gloves":1}
NEG = {"head_cap":5, "mask":6, "gloves":4}
DEV, MID = "cuda", "facebook/dinov2-base"
CROPS = Path("artifacts/crops")
root = Path("data/dump/dataset/apd_v3/dataset")
imgs = sorted([p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])

gt = {"images":[], "annotations":[], "categories":[{"id":i,"name":n} for i,n in enumerate(NAMES)]}
aid = 1
for iid,p in enumerate(imgs):
    with Image.open(p) as im: W,H = im.size
    gt["images"].append({"id":iid,"file_name":p.name,"width":W,"height":H})
    lp = root/"labels/val"/(p.stem+".txt")
    for line in (lp.read_text().splitlines() if lp.exists() else []):
        f=line.split()
        if len(f)<5: continue
        c=int(f[0]); cx,cy,bw,bh=(float(v) for v in f[1:5])
        gt["annotations"].append({"id":aid,"image_id":iid,"category_id":c,
            "bbox":[(cx-bw/2)*W,(cy-bh/2)*H,bw*W,bh*H],"area":bw*W*bh*H,"iscrowd":0}); aid+=1
Path("artifacts/_prod_val.json").write_text(json.dumps(gt))
with contextlib.redirect_stdout(io.StringIO()): coco=COCO("artifacts/_prod_val.json")
print(f"{len(imgs)} val images, {len(gt['annotations'])} GT boxes\n")

proc=AutoImageProcessor.from_pretrained(MID); enc=AutoModel.from_pretrained(MID).to(DEV).eval()
@torch.no_grad()
def embed(ims,b=64):
    o=[]
    for i in range(0,len(ims),b):
        o.append(enc(**proc(images=ims[i:i+b],return_tensors="pt").to(DEV)).last_hidden_state[:,0].float().cpu().numpy())
    return np.concatenate(o) if o else np.zeros((0,768),np.float32)

sp=json.loads(Path("artifacts/splits.json").read_text())["splits"]
fs={f:k for k,v in sp.items() for f in v}
idx=[json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]
hc={r["person_id"]:r["crop"] for r in idx if r["part"]=="head"}
hb={(r["person_id"],int(r["crop"].rsplit("_h",1)[1].split(".")[0])):r["crop"] for r in idx if r["part"]=="hand"}
rows={k:([],[]) for k in ("head_cap","mask","gloves")}
for r in (json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()):
    if fs.get(r["frame"])!="train": continue
    for k in ("head_cap","mask"):
        v=r["labels"][k]
        if v!=Value.UNCLEAR.value and r["person_id"] in hc:
            rows[k][0].append(hc[r["person_id"]]); rows[k][1].append(int(v=="present"))
    for h in r.get("hands",[]):
        c=hb.get((r["person_id"],h["hand"]))
        if c and h["value"]!=Value.UNCLEAR.value:
            rows["gloves"][0].append(c); rows["gloves"][1].append(int(h["value"]=="present"))
for m in (json.loads(l) for l in Path("artifacts/mined_labels.jsonl").read_text().splitlines()):
    if m["attribute"]=="mask" and m.get("split")=="train" and m["value"]!="unclear" and m["person_id"] in hc:
        rows["mask"][0].append(hc[m["person_id"]]); rows["mask"][1].append(int(m["value"]=="present"))
clf={k:LogisticRegression(max_iter=3000,C=0.05,class_weight="balanced").fit(
       embed([Image.open(CROPS/p).convert("RGB") for p in ps]),np.array(ys)) for k,(ps,ys) in rows.items()}
print("probes (fitted on SPPG crops, a different domain):",
      {k:f"{int(sum(y))}+/{len(y)-int(sum(y))}-" for k,(_,y) in rows.items()}, "\n")

pose=YOLO(".work/yolo11x-pose.pt")
def crop_of(img,box,size):
    x1,y1,x2,y2=box; pad=0.35*max(x2-x1,y2-y1)
    c=img[max(0,int(y1-pad)):int(y2+pad),max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else Image.fromarray(cv2.resize(c,(size,size)))
def iou(u,v):
    ax,ay,aw,ah=u; bx,by,bw,bh=v
    ix,iy=max(ax,bx),max(ay,by); jx,jy=min(ax+aw,bx+bw),min(ay+ah,by+bh)
    inter=max(0,jx-ix)*max(0,jy-iy); un=aw*ah+bw*bh-inter
    return inter/un if un>0 else 0.

pdets=[]; verdicts={}
for iid,p in enumerate(imgs):
    bgr=cv2.imread(str(p)); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    r=pose.predict(bgr,imgsz=GEOMETRY.pose_imgsz,conf=0.15,verbose=False,device=0)[0]
    if r.keypoints is None or not len(r.boxes): verdicts[iid]=[]; continue
    kp=r.keypoints.data.cpu().numpy(); bx=r.boxes.xyxy.cpu().numpy(); cf=r.boxes.conf.cpu().numpy()
    people=build_rois(bx,kp); frs=[frontality(k) for k in kp]
    for b,c in zip(bx,cf):
        pdets.append({"image_id":iid,"category_id":0,
            "bbox":[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])],"score":float(c)})
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
            per[i]["head_cap"]=clf["head_cap"].predict_proba(e)[0,1]
            if frs[i]>=0.5: per[i]["mask"]=clf["mask"].predict_proba(e)[0,1]
        else: per[i].setdefault("h",[]).append(clf["gloves"].predict_proba(e)[0,1])
    out=[]
    for i,pp in enumerate(people):
        d=per[i]; v={}
        v["head_cap"]="unclear" if "head_cap" not in d else ("present" if d["head_cap"]>=.5 else "absent")
        v["mask"]="unclear" if "mask" not in d else ("present" if d["mask"]>=.5 else "absent")
        hs=d.get("h",[])
        v["gloves"]="unclear" if len(hs)<2 else ("present" if all(x>=.5 for x in hs) else "absent")
        out.append((pp.person_box,v))
    verdicts[iid]=out

with contextlib.redirect_stdout(io.StringIO()):
    e=COCOeval(coco,coco.loadRes(pdets),"bbox"); e.evaluate(); e.accumulate()
pr=e.eval["precision"][0,:,0,0,2]; pa=e.eval["precision"][:,:,0,0,2]
ngt=len(coco.getAnnIds(catIds=[0]))
d_op=[d for d in pdets if d["score"]>=0.30]; tp=0; used=collections.defaultdict(set)
gts={i:[coco.anns[x]["bbox"] for x in coco.getAnnIds(imgIds=[i],catIds=[0])] for i in range(len(imgs))}
for d in sorted(d_op,key=lambda x:-x["score"]):
    best,bi=0,None
    for j,g in enumerate(gts[d["image_id"]]):
        if j in used[d["image_id"]]: continue
        v=iou(d["bbox"],g)
        if v>best: best,bi=v,j
    if best>=0.5: tp+=1; used[d["image_id"]].add(bi)
print("PERSON DETECTION on the production val split")
print(f"  {'model':<24}{'GT':>5}{'AP@.5':>8}{'AP@.5:.95':>11}{'P@.3':>7}{'R@.3':>7}")
print(f"  {'ours (yolo11x-pose)':<24}{ngt:>5}{float(np.mean(pr[pr>-1])):>8.3f}"
      f"{float(np.mean(pa[pa>-1])):>11.3f}{tp/max(len(d_op),1):>7.2f}{tp/ngt:>7.2f}")
print(f"  {'apd_mbg (production)':<24}{198:>5}{0.959:>8.3f}{0.724:>11.3f}{0.87:>7.2f}{0.96:>7.2f}")

print(f"\nATTRIBUTES -- cross-domain, probes trained on SPPG crops")
print(f"  {'attribute':<12}{'matched':>9}{'decided':>9}{'agree':>7}{'rate':>7}{'majority':>10}")
for k in ("head_cap","mask","gloves"):
    tot=dec=ok=0; truths=[]
    for iid in range(len(imgs)):
        gp=[coco.anns[x]["bbox"] for x in coco.getAnnIds(imgIds=[iid],catIds=[0])]
        gpos=[coco.anns[x]["bbox"] for x in coco.getAnnIds(imgIds=[iid],catIds=[POS[k]])]
        gneg=[coco.anns[x]["bbox"] for x in coco.getAnnIds(imgIds=[iid],catIds=[NEG[k]])]
        for gb in gp:
            inb=lambda q: gb[0]<=q[0]+q[2]/2<=gb[0]+gb[2] and gb[1]<=q[1]+q[3]/2<=gb[1]+gb[3]
            hasp=any(inb(q) for q in gpos); hasn=any(inb(q) for q in gneg)
            if hasp==hasn: continue                     # no evidence, or contradictory
            truth="present" if hasp else "absent"
            best,bv=0,None
            for pb,v in verdicts.get(iid,[]):
                s=iou([pb[0],pb[1],pb[2]-pb[0],pb[3]-pb[1]],gb)
                if s>best: best,bv=s,v
            if best<0.5 or bv is None: continue
            tot+=1; truths.append(truth)
            if bv[k]!="unclear":
                dec+=1; ok += (bv[k]==truth)
    maj=max(truths.count("present"),truths.count("absent"))/max(len(truths),1) if truths else 0
    print(f"  {k:<12}{tot:>9}{dec:>9}{ok:>7}{(100*ok/dec if dec else float('nan')):>6.0f}%{100*maj:>9.0f}%")
print("\n  'majority' = always guessing the commoner class on the same matched persons.")
