"""Production vs ours on the same footage, one label design, current models.

Rebuilt from scripts/22, which fitted its probes inline from the SPPG seed
labels. Those are superseded: the pipeline now uses the ci3 classifiers and the
voter3_ci3 probes, mask abstains on a confidence margin rather than a frontality
gate, and frames are cleaned of the production dashboard's burned-in verdict so
that its own overlay cannot be mistaken for either model's output.

Both panels show per-person verdicts, since that is the decision the system
actually has to make. apd_mbg's PPE boxes are resolved to persons the way the
deployed service does it -- positive class vs negative class, centre inside the
person box. Verdict cards are placed to avoid covering one another.
"""
import argparse, collections, pickle, sys, time
from pathlib import Path
import cv2, numpy as np, torch, openvino as ov
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
from apd.overlay import clean
from apd.pose import build_rois, roi_usable
from apd.schema import GEOMETRY, assert_contract

assert_contract()
LABELS = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
POS = {"head_cap":"head_cap","mask":"mask","gloves":"gloves"}
NEG = {"head_cap":"no_head_cap","mask":"no_mask","gloves":"no_gloves"}
OK, BAD, UNK = (80,200,120), (60,60,235), (150,155,160)
ATTRS = ("head_cap","mask","gloves")
MASK_MARGIN = 0.25

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", default="artifacts/compare.mp4")
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--pose-imgsz", type=int, default=1280)
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--edge-pose", default="yolo11s-pose",
                help="third panel: the deployable configuration -- a small pose anchor "
                     "plus voter4 alone, no DINOv2 and no second voter.")
ap.add_argument("--cls-model", default="effnet_b1", choices=["effnet_b1", "effnet_b0", "yolo11n", "yolo11s", "yolo11m"],
                help="classifier architecture for edge panel")
ap.add_argument("--band-absent", type=float, default=0.10, help="lower confidence threshold for absent")
ap.add_argument("--band-present", type=float, default=0.90, help="upper confidence threshold for present")
ap.add_argument("--seconds", type=float, default=10.0,
                help="length of clip to render; 0 for the whole file")
ap.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
ap.add_argument("--clean", type=int, default=0,
                help="erase a production dashboard overlay from each frame.")
a = ap.parse_args()
DEV = "cuda"

core = ov.Core(); ovdev = "GPU" if "GPU" in core.available_devices else "CPU"
mbg = core.compile_model(core.read_model("model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml"), ovdev)
mbg_out = mbg.outputs[0]
pose = YOLO(".work/yolo11x-pose.pt")
probes = pickle.load(open("artifacts/voter3_inpaint/probes.pkl","rb"))
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
enc = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
y26 = {k: YOLO(f"runs/classify/runs/classify_inpaint/{k}/weights/best.pt") for k in ATTRS}

from torchvision import transforms, models

effnet_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

effnet_models = {}
if "effnet" in a.cls_model:
    for k in ATTRS:
        ckpt_path = Path(f"runs/classify_{a.cls_model}/{k}_best.pt")
        if not ckpt_path.exists():
            ckpt_path = Path(f"runs/classify_effnet_b1/{k}_best.pt")
        if a.cls_model == "effnet_b0":
            m = models.efficientnet_b0()
        else:
            m = models.efficientnet_b1()
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
        m.load_state_dict(torch.load(ckpt_path, map_location=DEV))
        m = m.to(DEV).eval()
        effnet_models[k] = m

def predict_crop_prob(crop_bgr, attr_name):
    if "effnet" in a.cls_model and attr_name in effnet_models:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = effnet_transform(pil_img).unsqueeze(0).to(DEV)
        with torch.no_grad():
            out = effnet_models[attr_name](tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
        return prob
    else:
        res = y26[attr_name].predict(crop_bgr, verbose=False, device=DEV)[0]
        pres_idx = {v:i for i,v in y26[attr_name].names.items()}["present"]
        return float(res.probs.data[pres_idx].cpu().numpy())

def apply_confidence_band(prob, t_absent=a.band_absent, t_present=a.band_present):
    if prob >= t_present:
        return "present"
    elif prob <= t_absent:
        return "absent"
    else:
        return "unclear"

# --- the distilled student: one shared backbone over the same ROIs ---------
edge_pose = YOLO(f".work/{a.edge_pose}.pt" if Path(f".work/{a.edge_pose}.pt").exists()
                 else f"{a.edge_pose}.pt")
if "effnet" in a.cls_model and effnet_models:
    cls_params = sum(sum(p.numel() for p in m.parameters()) for m in effnet_models.values())
else:
    cls_params = sum(sum(p.numel() for p in m.model.parameters()) for m in y26.values())
edge_params = sum(p.numel() for p in edge_pose.model.parameters()) + cls_params
y26i = {k: {v:i for i,v in m.names.items()} for k,m in y26.items()}

@torch.no_grad()
def embed(ims):
    if not ims: return np.zeros((0,768), np.float32)
    return enc(**proc(images=ims, return_tensors="pt").to(DEV)).last_hidden_state[:,0].float().cpu().numpy()

def letterbox(img, s=640):
    h,w = img.shape[:2]; r = min(s/h, s/w); nh,nw = round(h*r), round(w*r)
    o = np.full((s,s,3),114,np.uint8); t,l = (s-nh)//2,(s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img,(nw,nh)); return o,r,l,t

def crop_np(img, box, size):
    x1,y1,x2,y2 = box; pad = 0.35*max(x2-x1, y2-y1)
    c = img[max(0,int(y1-pad)):int(y2+pad), max(0,int(x1-pad)):int(x2+pad)]
    return None if c.size==0 or min(c.shape[:2])<8 else cv2.resize(c,(size,size))

def centre_in(b, p):
    cx,cy = (b[0]+b[2])/2,(b[1]+b[3])/2
    return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]

def draw(panel, people, title, sub):
    """Cards placed clear of one another; a leader line keeps each with its person."""
    H_, W_ = panel.shape[:2]
    CW, CH = 92, 46
    placed = []
    def free(x, y):
        if x < 2 or y < 28 or x+CW > W_-2 or y+CH > H_-2: return False
        return all(not (x < px+CW and x+CW > px and y < py+CH and y+CH > py) for px,py in placed)
    for pbox, v in sorted(people, key=lambda t: t[0][0]):
        x1,y1,x2,y2 = [int(t) for t in pbox]
        col = BAD if "absent" in v.values() else (OK if "present" in v.values() else UNK)
        cv2.rectangle(panel,(x1,y1),(x2,y2),col,2)
        for cx,cy in ((x1,y1-CH-4),(x2+4,y1),(x1-CW-4,y1),(x1,y2+4),(x2+4,y2-CH),(x1,y1+4)):
            if free(cx,cy): break
        else:
            cx,cy = min(max(2,x1), W_-CW-2), 28
            while not free(cx,cy) and cy+CH < H_-2: cy += CH+3
        placed.append((cx,cy))
        cv2.line(panel,(x1+2,y1+2),(cx+CW//2,cy+CH//2),col,1)
        cv2.rectangle(panel,(cx,cy),(cx+CW,cy+CH),(18,24,28),-1)
        cv2.rectangle(panel,(cx,cy),(cx+CW,cy+CH),col,1)
        for j,k in enumerate(ATTRS):
            s = v[k]; mark = "OK" if s=="present" else ("X" if s=="absent" else "?")
            c2 = OK if s=="present" else (BAD if s=="absent" else UNK)
            cv2.putText(panel,f"{mark} {k}",(cx+5,cy+14+13*j),cv2.FONT_HERSHEY_SIMPLEX,0.36,c2,1)
    cv2.rectangle(panel,(0,0),(W_,26),(18,24,28),-1)
    cv2.putText(panel,title,(8,18),cv2.FONT_HERSHEY_SIMPLEX,0.52,(235,240,242),1)
    (tw,_),_ = cv2.getTextSize(sub,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
    cv2.putText(panel,sub,(max(8,W_-tw-8),18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,200,220),1)
    return panel

cap = cv2.VideoCapture(a.video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 25
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS/a.stride, (W*3, H))
print(f"{a.video}\n{W}x{H} @ {FPS:.1f} -> {W*3}x{H} @ {FPS/a.stride:.1f}   {a.out}")

if a.start: cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start*FPS))
max_src = int(a.seconds*FPS) if a.seconds else 0
n = written = 0; t0 = time.time()
stat = collections.defaultdict(collections.Counter); pc = {"a":[], "b":[], "c":[]}
while True:
    ok, fr = cap.read()
    if not ok: break
    if n % a.stride:
        n += 1; continue
    if a.clean:
        fr = clean(fr)
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

    # ---- production model: PPE boxes resolved to persons
    img, r, dx, dy = letterbox(rgb)
    p = mbg([(img.astype(np.float32)/255).transpose(2,0,1)[None]])[mbg_out][0]
    if p.shape[0] < p.shape[1]: p = p.T
    sc = p[:,4:11]; cl = sc.argmax(1); cf = sc[np.arange(len(sc)), cl]; k = cf >= a.conf
    b, cf, cl = p[k][:,:4], cf[k], cl[k]; boxes = np.zeros((0,4))
    if len(b):
        xy = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2],1)
        xy[:,[0,2]] -= dx; xy[:,[1,3]] -= dy; xy /= r
        ii = cv2.dnn.NMSBoxes([[float(v[0]),float(v[1]),float(v[2]-v[0]),float(v[3]-v[1])] for v in xy],
                              cf.tolist(), a.conf, 0.7)
        ii = np.array(ii).flatten() if len(ii) else np.zeros(0,int); boxes, cl = xy[ii], cl[ii]
    pboxes = [boxes[i] for i in range(len(cl)) if cl[i] == 0]
    ppe = [(boxes[i], LABELS[cl[i]]) for i in range(len(cl)) if cl[i] != 0]
    A = []
    for pb in pboxes:
        v = {}
        for at in ATTRS:
            hp = any(nm==POS[at] and centre_in(bb, pb) for bb,nm in ppe)
            hn = any(nm==NEG[at] and centre_in(bb, pb) for bb,nm in ppe)
            v[at] = "absent" if hn else ("present" if hp else "unclear")
            stat["a"][f"{at}:{v[at]}"] += 1
        A.append((pb, v))
    pc["a"].append(len(pboxes))

    # ---- ours: pose anchors, two voters, agreement required
    rr = pose.predict(fr, imgsz=a.pose_imgsz, conf=0.40, verbose=False, device=0)[0]
    B, C = [], []
    if rr.keypoints is not None and len(rr.boxes):
        people = build_rois(rr.boxes.xyxy.cpu().numpy(), rr.keypoints.data.cpu().numpy())
        for pp in people:
            _pb = [q.person_box for q in people]
            _ok = roi_usable(pp.head, fr.shape, part="head", others=_pb, self_box=pp.person_box)
            hc = crop_np(fr, pp.head, GEOMETRY.head_crop_px) if (_ok and not pp.head_collision) else None
            hands = [c_ for h in pp.hands[:2]
                     if roi_usable(h, fr.shape, part="hand", others=_pb, self_box=pp.person_box)
                     for c_ in [crop_np(fr, h, GEOMETRY.hand_crop_px)] if c_ is not None]
            v = {}
            e = embed([Image.fromarray(cv2.cvtColor(hc, cv2.COLOR_BGR2RGB))])[0] if hc is not None else None
            for at in ATTRS:
                if at == "gloves":
                    if len(hands) < 2: v[at] = "unclear"; continue
                    eh = embed([Image.fromarray(cv2.cvtColor(c_, cv2.COLOR_BGR2RGB)) for c_ in hands[:2]])
                    p3 = [float(probes["gloves"].predict_proba(x[None])[0,1]) for x in eh]
                    p4 = [float(y26["gloves"].predict(c_, verbose=False, device=0)[0]
                          .probs.data[y26i["gloves"]["present"]]) for c_ in hands[:2]]
                    a3 = "present" if all(x >= .5 for x in p3) else "absent"
                    a4 = "present" if all(x >= .5 for x in p4) else "absent"
                else:
                    if hc is None: v[at] = "unclear"; continue
                    q3 = float(probes[at].predict_proba(e[None])[0,1])
                    q4 = float(y26[at].predict(hc, verbose=False, device=0)[0].probs.data[y26i[at]["present"]])
                    if at == "mask" and (abs(q3-.5) < MASK_MARGIN or abs(q4-.5) < MASK_MARGIN):
                        v[at] = "unclear"; continue
                    a3 = "present" if q3 >= .5 else "absent"
                    a4 = "present" if q4 >= .5 else "absent"
                v[at] = a3 if a3 == a4 else "unclear"
            for at in ATTRS: stat["b"][f"{at}:{v[at]}"] += 1
            B.append((pp.person_box, v))


    pc["b"].append(len(B))

    # ---- edge candidate: its OWN small pose anchor, then voter4 alone.
    # Given its own anchors rather than the teacher's, because a smaller pose
    # model finding fewer people is part of what shipping it costs.
    er = edge_pose.predict(fr, imgsz=a.pose_imgsz, conf=0.40, verbose=False, device=0)[0]
    if er.keypoints is not None and len(er.boxes):
        for pp in build_rois(er.boxes.xyxy.cpu().numpy(), er.keypoints.data.cpu().numpy()):
            _pb = [q.person_box for q in people]
            _ok = roi_usable(pp.head, fr.shape, part="head", others=_pb, self_box=pp.person_box)
            hc = crop_np(fr, pp.head, GEOMETRY.head_crop_px) if (_ok and not pp.head_collision) else None
            hands = [c_ for h in pp.hands[:2]
                     if roi_usable(h, fr.shape, part="hand", others=_pb, self_box=pp.person_box)
                     for c_ in [crop_np(fr, h, GEOMETRY.hand_crop_px)] if c_ is not None]
            v = {}
            for at in ATTRS:
                if at == "gloves":
                    if len(hands) == 0:
                        v[at] = "unclear"
                        continue
                    ps = [predict_crop_prob(c_, "gloves") for c_ in hands[:2]]
                    h_decisions = [apply_confidence_band(p) for p in ps]
                    if all(d == "present" for d in h_decisions):
                        v[at] = "present"
                    elif any(d == "absent" for d in h_decisions):
                        v[at] = "absent"
                    else:
                        v[at] = "unclear"
                else:
                    if hc is None: v[at] = "unclear"; continue
                    if at == "mask" and not pp.frontal_face:
                        v[at] = "unclear"  # Abstain when person turned away from CCTV
                        continue
                    q = predict_crop_prob(hc, at)
                    v[at] = apply_confidence_band(q)
            for at in ATTRS: stat["c"][f"{at}:{v[at]}"] += 1
            C.append((pp.person_box, v))
    pc["c"].append(len(C))

    left = draw(fr.copy(), A, "production  apd_mbg (yolo11s)", f"persons: {len(A)}")
    mid = draw(fr.copy(), B, "teacher  pose -> ROI -> 2 voters", f"persons: {len(B)}")
    right = draw(fr.copy(), C, f"edge  {a.edge_pose} + {a.cls_model} ({a.band_absent:.2f}/{a.band_present:.2f})  ({edge_params/1e6:.1f}M)", f"persons: {len(C)}")
    wr.write(np.concatenate([left, mid, right], axis=1))
    written += 1; n += 1
    if written % 50 == 0:
        print(f"  {written} frames  {written/(time.time()-t0):.1f} fps", flush=True)
    if a.max_frames and written >= a.max_frames: break
    if max_src and n >= max_src: break

cap.release(); wr.release()
print(f"\n{written} frames in {time.time()-t0:.0f}s -> {a.out}")
print(f"mean persons/frame   production {np.mean(pc['a']):.2f}   "
      f"teacher {np.mean(pc['b']):.2f}   edge {np.mean(pc['c']):.2f}")
for side, name in (("a","production"), ("b","teacher"), ("c","edge")):
    print(f"\n{name} verdicts")
    for at in ATTRS:
        c = stat[side]
        tot = sum(c[f"{at}:{v}"] for v in ("present","absent","unclear")) or 1
        print(f"  {at:9} present {c[f'{at}:present']:5d}  absent {c[f'{at}:absent']:5d}  "
              f"unclear {c[f'{at}:unclear']:5d}  ({100*(1-c[f'{at}:unclear']/tot):.0f}% decided)")
