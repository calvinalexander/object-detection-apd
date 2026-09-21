"""Estimate the ID-switch rate without ground-truth identities.

True IDSW needs annotated identities, which we do not have, so this measures two
independent signatures of a switch and reports them as proxies:

  appearance jump   the crop inside a track suddenly looks like a different
                    person. Calibrated, not thresholded by eye: the SAME-person
                    scale comes from consecutive observations within a track,
                    and the DIFFERENT-person scale from crops of two tracks
                    visible in the same frame (necessarily different people).
                    A jump is counted when a within-track step looks more like
                    the different-person distribution than the same-person one.

  crossing          two boxes overlapping around the moment of the jump. A
                    switch has to happen somewhere: appearance changes that
                    coincide with a crossing are the ones worth believing.

Reported per crossing and per track-minute, because the number that matters for
temporal aggregation is how often a track absorbs a second person.
"""
import argparse, collections, sys, time
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.40)
ap.add_argument("--tracker", default="botsort.yaml")
ap.add_argument("--iou-cross", type=float, default=0.15)
a = ap.parse_args()

dev = "cuda"
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(dev).eval()
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

@torch.no_grad()
def embed(crops):
    if not crops: return np.zeros((0, 384), np.float32)
    b = []
    for c in crops:
        im = cv2.cvtColor(cv2.resize(c, (126, 126)), cv2.COLOR_BGR2RGB) / 255.0
        b.append(((im - MEAN) / STD).transpose(2, 0, 1))
    f = dino(torch.tensor(np.array(b), dtype=torch.float32, device=dev))
    return torch.nn.functional.normalize(f, dim=1).cpu().numpy()

def iou(a_, b_):
    ix = max(0, min(a_[2], b_[2]) - max(a_[0], b_[0]))
    iy = max(0, min(a_[3], b_[3]) - max(a_[1], b_[1]))
    inter = ix * iy
    ua = (a_[2]-a_[0])*(a_[3]-a_[1]) + (b_[2]-b_[0])*(b_[3]-b_[1]) - inter
    return inter / max(1e-6, ua)

pose = YOLO(".work/yolo11x-pose.pt")
cap = cv2.VideoCapture(a.video)
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
want = int(a.seconds * FPS / a.stride)

obs = collections.defaultdict(list)          # tid -> [(fidx, emb, box)]
frames_boxes = []                            # [ {tid: box} ]
n = kept = 0
t0 = time.time()
while kept < want:
    ok, fr = cap.read()
    if not ok: break
    if n % a.stride: n += 1; continue
    r = pose.track(fr, imgsz=a.imgsz, conf=a.conf, persist=True,
                   tracker=a.tracker, verbose=False, device=0)[0]
    cur = {}
    if r.boxes is not None and r.boxes.id is not None:
        ids = r.boxes.id.int().cpu().tolist()
        xy = r.boxes.xyxy.cpu().numpy()
        crops, keys = [], []
        for tid, bb in zip(ids, xy):
            x1, y1, x2, y2 = [int(v) for v in bb]
            c = fr[max(0, y1):y2, max(0, x1):x2]
            if c.size == 0 or min(c.shape[:2]) < 12: continue
            crops.append(c); keys.append((tid, (x1, y1, x2, y2)))
        E = embed(crops)
        for (tid, bb), e in zip(keys, E):
            obs[tid].append((kept, e, bb)); cur[tid] = bb
    frames_boxes.append(cur)
    kept += 1; n += 1
    if kept % 50 == 0: print(f"  {kept}/{want} frames", flush=True)
cap.release()

conc = [len(f) for f in frames_boxes]
print(f"\n{Path(a.video).name}")
print(f"{kept} frames ({kept*a.stride/FPS:.0f}s at stride {a.stride}), "
      f"persons/frame mean {np.mean(conc):.2f} max {max(conc) if conc else 0}")
print(f"distinct track ids: {len(obs)}   ids per max-concurrent: "
      f"{len(obs)/max(1,max(conc) if conc else 1):.2f}")

# ---- calibration: same-person vs different-person distance scales
same = []
for tid, seq in obs.items():
    for (f0, e0, _), (f1, e1, _) in zip(seq, seq[1:]):
        if f1 - f0 == 1: same.append(1.0 - float(e0 @ e1))
diff = []
for cur in frames_boxes:
    ids = list(cur)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ea = next((e for f, e, _ in obs[ids[i]] if f == frames_boxes.index(cur)), None)
    break                                    # replaced below by a cheaper pass
by_frame = collections.defaultdict(dict)
for tid, seq in obs.items():
    for f, e, bb in seq: by_frame[f][tid] = e
diff = []
rng = np.random.default_rng(0)
for f, d in by_frame.items():
    ids = list(d)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            diff.append(1.0 - float(d[ids[i]] @ d[ids[j]]))
same = np.array(same); diff = np.array(diff)
print(f"\nappearance distance  same person (consecutive, n={len(same)}): "
      f"median {np.median(same):.3f}  p95 {np.percentile(same,95):.3f}")
print(f"                     different people (same frame, n={len(diff)}): "
      f"median {np.median(diff):.3f}  p5  {np.percentile(diff,5):.3f}")
thr = float(np.percentile(diff, 5))
sep = float((same < thr).mean())
print(f"threshold = different-person p5 = {thr:.3f}   "
      f"({sep:.0%} of same-person steps fall below it, so the two scales "
      f"{'separate' if sep>0.8 else 'OVERLAP -- treat results as weak'})")

# ---- jumps, and whether they coincide with a crossing
jumps, jump_cross = 0, 0
steps = 0
for tid, seq in obs.items():
    for (f0, e0, b0), (f1, e1, b1) in zip(seq, seq[1:]):
        if f1 - f0 != 1: continue
        steps += 1
        if 1.0 - float(e0 @ e1) < thr: continue
        jumps += 1
        crossed = False
        for f in (f0, f1):
            for otid, ob in frames_boxes[f].items():
                if otid != tid and iou(b1, ob) > a.iou_cross:
                    crossed = True; break
            if crossed: break
        jump_cross += crossed

cross_events = 0
for f, cur in enumerate(frames_boxes):
    ids = list(cur)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if iou(cur[ids[i]], cur[ids[j]]) > a.iou_cross: cross_events += 1

track_minutes = sum(len(s) for s in obs.values()) * a.stride / FPS / 60
print(f"\n{steps} consecutive-frame steps across {len(obs)} tracks "
      f"({track_minutes:.1f} track-minutes)")
print(f"appearance jumps                 : {jumps} ({jumps/max(1,steps):.2%} of steps)")
print(f"  of which during an overlap     : {jump_cross}")
print(f"overlap events (frame-pairs)     : {cross_events}")
print(f"estimated switches per track-min : {jump_cross/max(1e-9,track_minutes):.2f}")
print(f"estimated switch rate per overlap: {jump_cross/max(1,cross_events):.3%}")
frag = collections.Counter(len(s) for s in obs.values())
short = sum(v for k, v in frag.items() if k <= 3)
print(f"tracks lasting <=3 frames        : {short}/{len(obs)} (fragmentation)")
