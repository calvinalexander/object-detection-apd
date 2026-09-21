"""How much detected PPE has no person box to attach to?

The deployed service assigns PPE to person *tracks* (assign_apd_smart). Any PPE
box that falls outside every person box is silently dropped -- no track, no
event. This measures that loss using the deployed model's own output.
"""
import argparse, collections
from pathlib import Path
import cv2, numpy as np, openvino as ov

LABELS = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--video", required=True)
ap.add_argument("--conf", type=float, default=0.30); ap.add_argument("--nms", type=float, default=0.7)
ap.add_argument("--stride", type=int, default=5)
args = ap.parse_args()

core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
compiled = core.compile_model(core.read_model(args.model), dev); port = compiled.outputs[0]

def letterbox(img, s=640):
    h, w = img.shape[:2]; r = min(s/h, s/w); nh, nw = round(h*r), round(w*r)
    out = np.full((s,s,3),114,np.uint8); t,l = (s-nh)//2,(s-nw)//2
    out[t:t+nh, l:l+nw] = cv2.resize(img,(nw,nh)); return out, r, l, t

def contains(inner, outer, frac=0.6):
    """Is the PPE box mostly inside the person box? Matches the spirit of
    assign_apd_smart's centre-inside test, but measured by area overlap."""
    x1,y1 = max(inner[0],outer[0]), max(inner[1],outer[1])
    x2,y2 = min(inner[2],outer[2]), min(inner[3],outer[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a = max(1e-6,(inner[2]-inner[0])*(inner[3]-inner[1]))
    return inter/a >= frac

cap = cv2.VideoCapture(args.video)
orphan, attached, n, frames = collections.Counter(), collections.Counter(), 0, 0
while True:
    ok, fr = cap.read()
    if not ok: break
    if n % args.stride: n += 1; continue
    img, r, dx, dy = letterbox(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    p = compiled([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
    if p.shape[0] < p.shape[1]: p = p.T
    sc = p[:,4:11]; cls = sc.argmax(1); conf = sc[np.arange(len(sc)),cls]
    k = conf >= args.conf
    b, conf, cls = p[k][:,:4], conf[k], cls[k]
    if len(b):
        xy = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2],1)
        xy[:,[0,2]] -= dx; xy[:,[1,3]] -= dy; xy /= r
        idx = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                               conf.tolist(), args.conf, args.nms)
        idx = np.array(idx).flatten() if len(idx) else np.zeros(0,int)
        xy, cls = xy[idx], cls[idx]
        persons = [xy[i] for i in range(len(cls)) if cls[i]==0]
        for i in range(len(cls)):
            if cls[i]==0: continue
            name = LABELS[cls[i]]
            (attached if any(contains(xy[i],pb) for pb in persons) else orphan)[name] += 1
    frames += 1; n += 1
cap.release()

print(f"{frames} frames sampled (every {args.stride}th)\n")
print(f"{'PPE class':<14}{'attached':>10}{'orphaned':>10}{'orphan rate':>13}")
tot_a = tot_o = 0
for c in LABELS[1:]:
    a, o = attached[c], orphan[c]; tot_a += a; tot_o += o
    if a+o: print(f"{c:<14}{a:>10}{o:>10}{100*o/(a+o):>12.0f}%")
print(f"{'ALL':<14}{tot_a:>10}{tot_o:>10}{100*tot_o/max(tot_a+tot_o,1):>12.0f}%")
print("\nOrphaned PPE has no person track to attach to, so assign_apd_smart")
print("discards it: those detections can never become a violation event.")
