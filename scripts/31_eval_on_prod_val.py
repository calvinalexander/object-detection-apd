"""Evaluate apd_mbg against the production validation split.

This is the split the deployed models were validated on, so unlike the 53-frame
evaluation there is no contamination question and no camera-leakage guesswork.
It also scores all 7 classes -- the earlier run had to discard 1,109 `no_*`
detections because our annotation has no negatives to compare them against.

YOLO-format labels: class cx cy w h, normalised. Converted to COCO for scoring.
"""
import argparse, collections, contextlib, io, json, sys
from pathlib import Path
import cv2, numpy as np, openvino as ov
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dump/dataset/apd_v3/dataset")
ap.add_argument("--model", default="model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml")
ap.add_argument("--op", type=float, default=0.30)
ap.add_argument("--nms", type=float, default=0.7)
a = ap.parse_args()

root = Path(a.data)
imgs = sorted([p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])
print(f"{len(imgs)} val images from {root}")

gt = {"images":[], "annotations":[], "categories":[{"id":i,"name":n} for i,n in enumerate(NAMES)]}
aid = 1
for iid, p in enumerate(imgs):
    with Image.open(p) as im: W,H = im.size
    gt["images"].append({"id":iid,"file_name":p.name,"width":W,"height":H})
    lp = root/"labels/val"/(p.stem+".txt")
    if not lp.exists(): continue
    for line in lp.read_text().splitlines():
        f = line.split()
        if len(f) < 5: continue
        c = int(f[0]); cx,cy,bw,bh = (float(v) for v in f[1:5])
        x,y,w_,h_ = (cx-bw/2)*W, (cy-bh/2)*H, bw*W, bh*H
        gt["annotations"].append({"id":aid,"image_id":iid,"category_id":c,
                                  "bbox":[x,y,w_,h_],"area":w_*h_,"iscrowd":0}); aid += 1
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/_prod_val.json").write_text(json.dumps(gt))
with contextlib.redirect_stdout(io.StringIO()): coco = COCO("artifacts/_prod_val.json")
print(f"{len(gt['annotations'])} ground-truth boxes\n")

core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
net = core.compile_model(core.read_model(a.model), dev); port = net.outputs[0]

def letterbox(img, s=640):
    h,w = img.shape[:2]; r = min(s/h, s/w); nh,nw = round(h*r), round(w*r)
    o = np.full((s,s,3),114,np.uint8); t,l = (s-nh)//2,(s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img,(nw,nh)); return o,r,l,t

dets = []
for iid, p in enumerate(imgs):
    bgr = cv2.imread(str(p))
    img,r,dx,dy = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = net([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
    if out.shape[0] < out.shape[1]: out = out.T
    sc = out[:,4:11]; cl = sc.argmax(1); cf = sc.max(1)
    k = cf >= 0.001; b,cf,cl = out[k][:,:4], cf[k], cl[k]
    if not len(b): continue
    xy = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2],1)
    xy[:,[0,2]] -= dx; xy[:,[1,3]] -= dy; xy /= r
    idx = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                            cf.tolist(), 0.001, a.nms)
    idx = np.array(idx).flatten() if len(idx) else np.zeros(0,int)
    for i in idx:
        x1,y1,x2,y2 = xy[i]
        dets.append({"image_id":iid,"category_id":int(cl[i]),
                     "bbox":[float(x1),float(y1),float(x2-x1),float(y2-y1)],"score":float(cf[i])})
print(f"detections: {len(dets)}\n")

with contextlib.redirect_stdout(io.StringIO()):
    e = COCOeval(coco, coco.loadRes(dets), "bbox"); e.evaluate(); e.accumulate()

def iou(u,v):
    ax,ay,aw,ah=u; bx,by,bw,bh=v
    ix,iy=max(ax,bx),max(ay,by); jx,jy=min(ax+aw,bx+bw),min(ay+ah,by+bh)
    inter=max(0,jx-ix)*max(0,jy-iy); un=aw*ah+bw*bh-inter
    return inter/un if un>0 else 0.

print(f"apd_mbg on the production val split   (P/R at the deployed threshold {a.op})")
print(f"  {'class':<14}{'GT':>5}{'AP@.5':>8}{'AP@.5:.95':>11}{'P@op':>7}{'R@op':>7}")
aps=[]
for c,n in enumerate(NAMES):
    pr = e.eval["precision"][0,:,c,0,2]; ap50 = float(np.mean(pr[pr>-1])) if (pr>-1).any() else 0.
    pa = e.eval["precision"][:,:,c,0,2]; apa = float(np.mean(pa[pa>-1])) if (pa>-1).any() else 0.
    aps.append(ap50)
    ngt = len(coco.getAnnIds(catIds=[c]))
    d_op = [d for d in dets if d["category_id"]==c and d["score"]>=a.op]
    gts = {i:[coco.anns[x]["bbox"] for x in coco.getAnnIds(imgIds=[i],catIds=[c])] for i in range(len(imgs))}
    tp=0; used=collections.defaultdict(set)
    for d in sorted(d_op,key=lambda x:-x["score"]):
        best,bi=0,None
        for j,g in enumerate(gts[d["image_id"]]):
            if j in used[d["image_id"]]: continue
            v=iou(d["bbox"],g)
            if v>best: best,bi=v,j
        if best>=0.5: tp+=1; used[d["image_id"]].add(bi)
    P = tp/len(d_op) if d_op else float("nan"); R = tp/ngt if ngt else float("nan")
    print(f"  {n:<14}{ngt:>5}{ap50:>8.3f}{apa:>11.3f}{P:>7.2f}{R:>7.2f}")
print(f"  {'mAP@.5':<14}{'':>5}{np.mean(aps):>8.3f}")
print(f"\n  for contrast, on OUR 53 frames: person .470  head_cap .426  gloves .015  mask .050")
