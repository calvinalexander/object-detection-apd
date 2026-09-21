"""Evaluate the production detector against the 53 human-annotated frames.

Ground truth is positives-only (person, gloves, head_cap, mask), so the model's
three negative classes have nothing to score against and are excluded rather than
counted as false positives -- a `no_mask` box is not an error, it is a prediction
the annotation simply does not cover.

Reported per class and split by camera resolution, because the two populations
differ by ~4x in linear scale and an aggregate number hides that.
"""
import argparse, collections, json, sys, contextlib, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np, openvino as ov
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MODEL_LABELS = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
GT_CAT = {"person":3, "gloves":4, "head_cap":5, "mask":6}
EVAL = ["person","head_cap","gloves","mask"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml")
ap.add_argument("--conf", type=float, default=0.001)
ap.add_argument("--op", type=float, default=0.30)
ap.add_argument("--nms", type=float, default=0.7)
args = ap.parse_args()

ROOT = Path("data/sppg")
raw = json.loads((ROOT/".exports/coco-1778210446.4251382.json").read_text())
ann_imgs = {a["image_id"] for a in raw["annotations"]}
sub = {"images":[i for i in raw["images"] if i["id"] in ann_imgs],
       "annotations":[dict(a, iscrowd=0) for a in raw["annotations"]],
       "categories": raw["categories"]}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/_gt_53.json").write_text(json.dumps(sub))
with contextlib.redirect_stdout(io.StringIO()):
    coco = COCO("artifacts/_gt_53.json")
imgs = {i["id"]: i for i in sub["images"]}
print(f"ground truth: {len(sub['images'])} frames, {len(sub['annotations'])} boxes")

core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
net = core.compile_model(core.read_model(args.model), dev); port = net.outputs[0]
print(f"model: {Path(args.model).name} on {dev}")

def letterbox(img, s=640):
    h,w = img.shape[:2]; r = min(s/h, s/w); nh,nw = round(h*r), round(w*r)
    o = np.full((s,s,3),114,np.uint8); t,l = (s-nh)//2,(s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img,(nw,nh)); return o,r,l,t

dets, dropped = [], 0
for iid, im in imgs.items():
    bgr = cv2.imread(str(ROOT/im["file_name"]))
    img,r,dx,dy = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    p = net([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
    if p.shape[0] < p.shape[1]: p = p.T
    sc = p[:,4:11]; cl = sc.argmax(1); cf = sc[np.arange(len(sc)),cl]
    k = cf >= args.conf; b,cf,cl = p[k][:,:4], cf[k], cl[k]
    if not len(b): continue
    xy = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2],1)
    xy[:,[0,2]] -= dx; xy[:,[1,3]] -= dy; xy /= r
    keep = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                            cf.tolist(), args.conf, args.nms)
    keep = np.array(keep).flatten() if len(keep) else np.zeros(0,int)
    for i in keep:
        name = MODEL_LABELS[cl[i]]
        if name not in GT_CAT: dropped += 1; continue
        x1,y1,x2,y2 = xy[i]
        dets.append({"image_id": int(iid), "category_id": GT_CAT[name],
                     "bbox":[float(x1),float(y1),float(x2-x1),float(y2-y1)],
                     "score": float(cf[i])})
print(f"scored detections: {len(dets)}   negative-class boxes excluded: {dropped}\n")

def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ix,iy=max(ax,bx),max(ay,by); jx,jy=min(ax+aw,bx+bw),min(ay+ah,by+bh)
    inter=max(0,jx-ix)*max(0,jy-iy); u=aw*ah+bw*bh-inter
    return inter/u if u>0 else 0.0

def run(img_ids, tag):
    dt=[d for d in dets if d["image_id"] in img_ids]
    if not dt: print(f"{tag}: no detections\n"); return
    with contextlib.redirect_stdout(io.StringIO()):
        e=COCOeval(coco, coco.loadRes(dt), "bbox"); e.params.imgIds=list(img_ids)
        e.evaluate(); e.accumulate()
    cats=list(coco.getCatIds())
    print(f"{tag}  ({len(img_ids)} frames)")
    print(f"  {'class':<10}{'GT':>5}{'det@op':>8}{'AP@.5':>8}{'AP@.5:.95':>11}{'P@op':>7}{'R@op':>7}")
    for name in EVAL:
        cid=GT_CAT[name]; ci=cats.index(cid)
        pr=e.eval["precision"][0,:,ci,0,2]; ap50=float(np.mean(pr[pr>-1])) if (pr>-1).any() else float('nan')
        pa=e.eval["precision"][:,:,ci,0,2]; apa=float(np.mean(pa[pa>-1])) if (pa>-1).any() else float('nan')
        ngt=len(coco.getAnnIds(imgIds=list(img_ids), catIds=[cid]))
        d_op=[d for d in dt if d["category_id"]==cid and d["score"]>=args.op]
        gts={i:[coco.anns[a]["bbox"] for a in coco.getAnnIds(imgIds=[i],catIds=[cid])] for i in img_ids}
        tp=0; used=collections.defaultdict(set)
        for d in sorted(d_op,key=lambda x:-x["score"]):
            best,bi=0,None
            for j,g in enumerate(gts[d["image_id"]]):
                if j in used[d["image_id"]]: continue
                v=iou(d["bbox"],g)
                if v>best: best,bi=v,j
            if best>=0.5: tp+=1; used[d["image_id"]].add(bi)
        P=tp/len(d_op) if d_op else float('nan'); R=tp/ngt if ngt else float('nan')
        print(f"  {name:<10}{ngt:>5}{len(d_op):>8}{ap50:>8.3f}{apa:>11.3f}{P:>7.2f}{R:>7.2f}")
    print()

lo={i for i,v in imgs.items() if (v["width"],v["height"])==(704,576)}
hi={i for i,v in imgs.items() if (v["width"],v["height"])==(2880,1620)}
run(set(imgs),"ALL"); run(lo,"704x576  (low-res camera)"); run(hi,"2880x1620  (HD camera)")
print(f"P/R at the deployed threshold {args.op} (APD_GROUP_THRESH in config_apd.py); IoU 0.5")
