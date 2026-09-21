"""Both apd_mbg builds on the SDK's 442-image apd_v3 val set.

Person anchors are the ANNOTATED person boxes, not pose output, so this measures
the detector alone -- no pose model in the loop and nothing for it to blame. A
predicted PPE box is assigned to a person by centre-inside, the same rule the
deployed service uses.

Contamination warning, stated up front: the December build is described as
trained on `apd_v3_1065_ready`, and this is the apd_v3 val split. The March build
is trained on `apd_v4.9`, whose relationship to these images is unknown. Neither
number is a clean generalisation estimate; they are here to compare the builds.
"""
import argparse, collections, json, sys
from pathlib import Path
import cv2, numpy as np, openvino as ov
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
ATT = ["head_cap", "mask", "gloves"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="artifacts/sdk/sdk/customers/apd_v3/dataset")
ap.add_argument("--conf", type=float, default=0.25)     # config.yaml value
ap.add_argument("--iou", type=float, default=0.45)      # config.yaml value
ap.add_argument("--out", default="artifacts/prod_on_sdk.json")
a = ap.parse_args()

BUILDS = {
    "apd_mbg Dec-2025 (apd_v3_1065)": "model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml",
    "apd_mbg Mar-2026 (apd_v4.9)":    "artifacts/sdk/sdk/build/apd_mbg_openvino_model/best.xml",
}
core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"

def letterbox(img, s=640):
    h,w = img.shape[:2]; r = min(s/h, s/w); nh,nw = round(h*r), round(w*r)
    o = np.full((s,s,3),114,np.uint8); t,l = (s-nh)//2,(s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img,(nw,nh)); return o,r,l,t
def centre_in(b,p):
    cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

root = Path(a.data)
imgs = sorted(p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
print(f"{len(imgs)} images from {root}\n")

results = {}
for label, xml in BUILDS.items():
    if not Path(xml).exists():
        print(f"{label}: missing"); continue
    net = core.compile_model(core.read_model(xml), dev); port = net.outputs[0]
    rows = {at: [] for at in ATT}
    for n, p in enumerate(imgs, 1):
        im = cv2.imread(str(p))
        if im is None: continue
        H, W = im.shape[:2]
        gtb = []
        lp = root/"labels/val"/(p.stem+".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                f = line.split()
                if len(f) < 5: continue
                c = int(f[0]); cx,cy,bw,bh = (float(v) for v in f[1:5])
                gtb.append(([(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H], NAMES[c]))
        persons = [b for b,nm in gtb if nm == "person"]
        if not persons: continue

        img, r, dx, dy = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        o = net([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
        if o.shape[0] < o.shape[1]: o = o.T
        sc = o[:,4:11]; cl = sc.argmax(1); cf = sc.max(1); k = cf >= a.conf
        b, cf, cl = o[k][:,:4], cf[k], cl[k]; ppe = []
        if len(b):
            xy = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2],1)
            xy[:,[0,2]] -= dx; xy[:,[1,3]] -= dy; xy /= r
            idx = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                                   cf.tolist(), a.conf, a.iou)
            idx = np.array(idx).flatten() if len(idx) else np.zeros(0,int)
            ppe = [(xy[i], NAMES[cl[i]]) for i in idx if NAMES[cl[i]] != "person"]

        for pb in persons:
            for at in ATT:
                hp = any(nm==at and centre_in(bb, pb) for bb,nm in gtb)
                hn = any(nm==f"no_{at}" and centre_in(bb, pb) for bb,nm in gtb)
                if hp == hn: continue
                y = 0 if hp else 1                      # 1 = violation
                mp = any(nm==at and centre_in(bb, pb) for bb,nm in ppe)
                mn = any(nm==f"no_{at}" and centre_in(bb, pb) for bb,nm in ppe)
                pred = 0 if mp == mn else (0 if mp else 1)   # abstain -> no alert
                rows[at].append((y, pred, mp == mn))
        if n % 100 == 0: print(f"  {label}: {n}/{len(imgs)}", flush=True)

    print(f"\n=== {label} ===")
    print(f"  {'attr':9} {'n':>5} {'viol':>5} {'abstain':>8} {'prec':>7} {'recall':>7} {'F1':>7} {'MCC':>7} {'bal-acc':>8}")
    results[label] = {}
    for at in ATT:
        r_ = rows[at]
        if len(r_) < 10: continue
        y = np.array([t for t,_,_ in r_]); q = np.array([p for _,p,_ in r_])
        ab = sum(1 for _,_,x in r_ if x)
        pr, rc, f1, _ = precision_recall_fscore_support(y, q, average="binary", zero_division=0)
        mcc = matthews_corrcoef(y, q) if len(set(q)) > 1 else 0.0
        ba = balanced_accuracy_score(y, q)
        bp = y.mean()
        results[label][at] = dict(n=len(y), viol=int(y.sum()), abstain=ab, precision=pr,
                                  recall=rc, f1=f1, mcc=mcc, bal_acc=ba, base_rate=float(bp))
        print(f"  {at:9} {len(y):5d} {int(y.sum()):5d} {ab/len(y):7.0%} {pr:7.1%} {rc:7.1%} {f1:7.3f} {mcc:7.3f} {ba:8.1%}")
    print(f"  {'':9} {'':5} {'':5} {'':8} always-flag baseline F1: " +
          ", ".join(f"{at} {2*results[label][at]['base_rate']/(1+results[label][at]['base_rate']):.3f}"
                    for at in ATT if at in results[label]))
Path(a.out).write_text(json.dumps(results, indent=2))
print(f"\nwrote {a.out}")
