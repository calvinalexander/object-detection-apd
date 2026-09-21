"""Is unanimity actually right? Measured against the production annotation.

Three models agreeing 84% of the time says nothing about whether they are
correct -- they share a person detector, a domain and, until now, most of their
training data. This scores the consensus rule itself on the only split with
independent labels: the production val set, which annotates `no_*` explicitly,
so ABSENT is an assertion rather than an absent box.

Person anchors come from pose, as in deployment. A ground-truth verdict for a
person is derived the way the deployed service assigns PPE: a positive box whose
centre falls inside the person box means PRESENT, a `no_*` box means ABSENT,
neither means the annotator did not label that attribute for that person and the
person is skipped -- scoring an unlabelled person as a miss would invent errors.

Reported per confidence tier, because the point of the rule is that unanimity
should be trustworthy enough to accept unreviewed and disagreement should not.
"""
import argparse, collections, json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch, openvino as ov
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from apd.pose import build_rois, frontality
from apd.schema import GEOMETRY, Attribute, assert_contract

assert_contract()
NAMES = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
POS = {"head_cap":"head_cap","mask":"mask","gloves":"gloves"}
NEG = {"head_cap":"no_head_cap","mask":"no_mask","gloves":"no_gloves"}
ATTRS = ("head_cap","mask","gloves")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dump/dataset/apd_v3/dataset")
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--out", default="artifacts/consensus_validation.json")
a = ap.parse_args()

root = Path(a.data)
imgs = sorted(p for p in (root/"images/val").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
print(f"{len(imgs)} annotated val images")

core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
mbg = core.compile_model(core.read_model("model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml"), dev)
port = mbg.outputs[0]
pose = YOLO(".work/yolo11x-pose.pt")
probes = pickle.load(open("artifacts/voter3_ci3/probes.pkl","rb"))
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
enc = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()
y26 = {k: YOLO(f"runs/classify/runs/classify/runs/ci3_{k}/weights/best.pt") for k in ATTRS}
y26i = {k: {v:i for i,v in m.names.items()} for k,m in y26.items()}

@torch.no_grad()
def embed(ims):
    if not ims: return np.zeros((0,768), np.float32)
    return enc(**proc(images=ims, return_tensors="pt").to("cuda")).last_hidden_state[:,0].float().cpu().numpy()

def letterbox(img, s=640):
    h,w = img.shape[:2]; r = min(s/h, s/w); nh,nw = round(h*r), round(w*r)
    o = np.full((s,s,3),114,np.uint8); t,l = (s-nh)//2,(s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img,(nw,nh)); return o,r,l,t

def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

def centre_in(b, p):
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

res = collections.defaultdict(lambda: collections.Counter())
per_model = collections.defaultdict(lambda: collections.Counter())
for n, p in enumerate(imgs, 1):
    bgr = cv2.imread(str(p))
    if bgr is None: continue
    H, W = bgr.shape[:2]

    # ---- ground truth boxes
    lp = root/"labels/val"/(p.stem+".txt")
    gtb = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) < 5: continue
            c = int(f[0]); cx,cy,bw,bh = (float(v) for v in f[1:5])
            gtb.append(([(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H], NAMES[c]))

    rr = pose.predict(bgr, imgsz=1280, conf=0.40, verbose=False, device=0)[0]
    if rr.keypoints is None or not len(rr.boxes): continue
    kp = rr.keypoints.data.cpu().numpy()
    people = build_rois(rr.boxes.xyxy.cpu().numpy(), kp); frs = [frontality(k) for k in kp]

    img, r, dx, dy = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    o = mbg([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[port][0]
    if o.shape[0] < o.shape[1]: o = o.T
    sc = o[:,4:11]; cl = sc.argmax(1); cf = sc.max(1); k = cf >= a.conf
    b, cf, cl = o[k][:,:4], cf[k], cl[k]; ppe = []
    if len(b):
        xy = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2],1)
        xy[:,[0,2]] -= dx; xy[:,[1,3]] -= dy; xy /= r
        idx = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                               cf.tolist(), a.conf, 0.7)
        idx = np.array(idx).flatten() if len(idx) else np.zeros(0,int)
        ppe = [(xy[i], NAMES[cl[i]]) for i in idx if NAMES[cl[i]] != "person"]

    heads = [crop_np(bgr, pp.head, GEOMETRY.head_crop_px)
             if (pp.head is not None and not pp.head_collision) else None for pp in people]
    hands = [[crop_np(bgr, h, GEOMETRY.hand_crop_px) for h in pp.hands[:2]] for pp in people]

    hidx = [i for i,h in enumerate(heads) if h is not None]
    E = embed([Image.fromarray(cv2.cvtColor(heads[i], cv2.COLOR_BGR2RGB)) for i in hidx])
    b3 = collections.defaultdict(dict)
    for i, e in zip(hidx, E):
        b3[i]["head_cap"] = float(probes["head_cap"].predict_proba(e[None])[0,1])
        if frs[i] >= 0.5: b3[i]["mask"] = float(probes["mask"].predict_proba(e[None])[0,1])
    him, hmap = [], []
    for i, hl in enumerate(hands):
        for c_ in hl:
            if c_ is not None:
                him.append(Image.fromarray(cv2.cvtColor(c_, cv2.COLOR_BGR2RGB))); hmap.append(i)
    for i, e in zip(hmap, embed(him)):
        b3[i].setdefault("h", []).append(float(probes["gloves"].predict_proba(e[None])[0,1]))

    for i, pp in enumerate(people):
        pb = pp.person_box
        for at in ATTRS:
            hasp = any(nm==POS[at] and centre_in(bb, pb) for bb,nm in gtb)
            hasn = any(nm==NEG[at] and centre_in(bb, pb) for bb,nm in gtb)
            if hasp == hasn:              # unlabelled or contradictory for this person
                continue
            true = "present" if hasp else "absent"

            v = {}
            mp = any(nm==POS[at] and centre_in(bb, pb) for bb,nm in ppe)
            mn = any(nm==NEG[at] and centre_in(bb, pb) for bb,nm in ppe)
            v["apd_mbg"] = "unclear" if mp==mn else ("present" if mp else "absent")
            if at == "gloves":
                hs = b3[i].get("h", [])
                v["voter3"] = "unclear" if len(hs)<2 else ("present" if all(x>=.5 for x in hs) else "absent")
            else:
                v["voter3"] = "unclear" if at not in b3[i] else ("present" if b3[i][at]>=.5 else "absent")
            if at == "gloves":
                hp = [c_ for c_ in hands[i] if c_ is not None]
                if len(hp) < 2: v["voter4"] = "unclear"
                else:
                    gs = [float(y26["gloves"].predict(c_, verbose=False, device=0)[0]
                          .probs.data[y26i["gloves"]["present"]]) for c_ in hp]
                    v["voter4"] = "present" if all(g>=.5 for g in gs) else "absent"
            else:
                hc = heads[i]
                if hc is None or (at=="mask" and frs[i]<0.5): v["voter4"] = "unclear"
                else:
                    pr = float(y26[at].predict(hc, verbose=False, device=0)[0]
                               .probs.data[y26i[at]["present"]])
                    v["voter4"] = "present" if pr>=.5 else "absent"

            for m, vv in v.items():
                if vv != "unclear":
                    per_model[f"{at}:{m}"]["n"] += 1
                    per_model[f"{at}:{m}"]["ok"] += int(vv == true)

            committed = [x for x in v.values() if x != "unclear"]
            if len(committed) >= 2 and len(set(committed)) == 1:
                dec, kind = committed[0], f"unanimous_{len(committed)}"
            elif len(committed) >= 2:
                c = collections.Counter(committed); top, cnt = c.most_common(1)[0]
                dec, kind = top, ("majority" if cnt > len(committed)/2 else "tie")
            else:
                dec, kind = "unclear", "insufficient"
            res[f"{at}:{kind}"]["n"] += 1
            if dec != "unclear":
                res[f"{at}:{kind}"]["ok"] += int(dec == true)

            # apd_mbg was trained for this split, so any tier that requires its
            # agreement is close to circular here. Our two voters alone are the
            # only reading of this set that is not.
            ours = [v[m] for m in ("voter3", "voter4") if v[m] != "unclear"]
            if len(ours) == 2:
                k2 = "ours_agree" if len(set(ours)) == 1 else "ours_split"
                res[f"{at}:{k2}"]["n"] += 1
                if len(set(ours)) == 1:
                    res[f"{at}:{k2}"]["ok"] += int(ours[0] == true)
    if n % 40 == 0:
        print(f"  {n}/{len(imgs)}", flush=True)

print("\nconsensus rule vs production annotation")
print(f"{'attribute':10} {'tier':14} {'n':>5} {'correct':>9}")
for at in ATTRS:
    for kind in ("unanimous_3","unanimous_2","majority","tie","insufficient","ours_agree","ours_split"):
        c = res[f"{at}:{kind}"]
        if not c["n"]: continue
        acc = f"{c['ok']/c['n']:.1%}" if kind not in ("insufficient","ours_split") else "     -"
        print(f"{at:10} {kind:14} {c['n']:5d} {acc:>9}")

print(f"\nindividual voters (only where they committed)")
print(f"{'attribute':10} {'model':10} {'n':>5} {'correct':>9}")
for at in ATTRS:
    for m in ("apd_mbg","voter3","voter4"):
        c = per_model[f"{at}:{m}"]
        if c["n"]:
            print(f"{at:10} {m:10} {c['n']:5d} {c['ok']/c['n']:8.1%}")
Path(a.out).write_text(json.dumps(
    {"tiers": {k: dict(v) for k,v in res.items()},
     "voters": {k: dict(v) for k,v in per_model.items()}}, indent=2))
print(f"\nwrote {a.out}")
