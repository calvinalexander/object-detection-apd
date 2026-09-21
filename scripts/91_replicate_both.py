"""Both perception stages under the production tracker, vote and cooldown.

Same video, same ByteTracker settings, same vote window, same cooldown, same
event rule. The only thing that differs is what produces the per-person
verdicts:

    prod  apd_v3 7-class detector (OpenVINO copy) + assign_apd_smart
    ours  yolo11n pose -> anatomical ROI -> packed classifiers + assign_ours

That isolates the question the deployment actually asks: given identical
downstream logic, what reaches the database?
"""
import argparse, os, sys, time
from collections import deque, defaultdict
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/tray_counter/tray/tray_counter"
sys.path.insert(0, str(ROOT / "deploy/apd_ours")); sys.path.insert(0, str(LIVE))
for k in ("S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET",
          "API_BASE_URL", "API_KEY", "EDGE_DEVICE_ID", "SPPG_CODE"):
    os.environ.setdefault(k, "x")                      # apd_config refuses to import without these
os.environ.setdefault("APD_OURS_THREADS", "2")
# apd_config creates PROJECT_DIR/LOG_DIR at import; point them at a scratch dir
# so importing production's helpers here touches nothing the device owns.
_scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "apd_replicate"
_scratch.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PROJECT_DIR", str(_scratch))
os.environ.setdefault("APD_STORAGE_PATH", str(_scratch / "storage"))

ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="+", required=True)
ap.add_argument("--max-frames", type=int, default=250)
ap.add_argument("--target-fps", type=float, default=10.0)
ap.add_argument("--person-conf", type=float, default=0.40, help="pose conf, = config.yaml decodeyolopose")
ap.add_argument("--prod-conf", type=float, default=0.25, help="apd_v3 conf, = its config.yaml decodeyolo")
ap.add_argument("--prod-xml", default="artifacts/sdk/sdk/build/apd_mbg_openvino_model/best.xml")
a = ap.parse_args()

from types import SimpleNamespace
from apd_perception_ours import OursPerception, assign_ours
from apd_tracker.byte_tracker import BYTETracker
from apd_utils import build_class_maps, assign_apd_smart
import openvino as ov

ENV = {k.split("=")[0].strip(): k.split("=", 1)[1].split("#")[0].strip()
       for k in (LIVE / ".env").read_text().splitlines()
       if k.strip().startswith("APD_") and "=" in k}
VOTE_WINDOW = int(ENV.get("APD_VOTE_WINDOW", 5)); MIN_AGE = int(ENV.get("APD_MIN_AGE_FRAMES", 10))
RATIO = float(ENV.get("APD_MIN_VIOLATION_RATIO", 0.6)); COOLDOWN = int(ENV.get("APD_COOLDOWN_MS", 600000))
GROUPS = ["headcap", "gloves", "mask"]
GROUP_THRESH = {g: float(ENV.get(f"APD_THRESH_{g.upper()}", 0.3)) for g in GROUPS}
NAMES = {0: "person", 1: "gloves", 2: "head_cap", 3: "mask", 4: "no_gloves", 5: "no_head_cap", 6: "no_mask"}
APD_GROUPS = {"headcap": {"positive": ["head_cap"], "negative": ["no_head_cap"]},
              "mask": {"positive": ["mask"], "negative": ["no_mask"]},
              "gloves": {"positive": ["gloves"], "negative": ["no_gloves"]}}
CLASS_TO_GROUP = build_class_maps(NAMES, APD_GROUPS)

targs = SimpleNamespace(track_thresh=0.30, track_buffer=60, match_thresh=0.8,
                        aspect_ratio_thresh=1.6, min_box_area=10, mot20=False)
ours = OursPerception(SimpleNamespace(backend="ultralytics", pose_weights=str(ROOT / "yolo11n-pose.pt"),
                                      person_conf=a.person_conf, gate=1, imgsz=640, **vars(targs)))
core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
net = core.compile_model(core.read_model(a.prod_xml), dev); port = net.outputs[0]
print(f"prod model {Path(a.prod_xml).name} on {dev} | ours pose conf {a.person_conf}, prod conf {a.prod_conf}")
print(f"vote {VOTE_WINDOW} frames, ratio {RATIO}, min age {MIN_AGE}, cooldown {COOLDOWN} ms, {a.target_fps} fps\n")


def letterbox(img, s=640):
    h, w = img.shape[:2]; r = min(s/h, s/w); nh, nw = round(h*r), round(w*r)
    o = np.full((s, s, 3), 114, np.uint8); t, l = (s-nh)//2, (s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(img, (nw, nh)); return o, r, l, t


def prod_detect(frame):
    """apd_v3 -> (person boxes with score, ppe detections) as the monitor builds them."""
    img, r, dx, dy = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    o = net([(img.astype(np.float32)/255).transpose(2, 0, 1)[None]])[port][0]
    if o.shape[0] < o.shape[1]: o = o.T
    sc = o[:, 4:11]; cl = sc.argmax(1); cf = sc.max(1); k = cf >= a.prod_conf
    b, cf, cl = o[k][:, :4], cf[k], cl[k]
    if not len(b): return [], []
    xy = np.stack([b[:, 0]-b[:, 2]/2, b[:, 1]-b[:, 3]/2, b[:, 0]+b[:, 2]/2, b[:, 1]+b[:, 3]/2], 1)
    xy[:, [0, 2]] -= dx; xy[:, [1, 3]] -= dy; xy /= r
    idx = cv2.dnn.NMSBoxes([[float(v[0]), float(v[1]), float(v[2]-v[0]), float(v[3]-v[1])] for v in xy],
                           cf.tolist(), a.prod_conf, 0.45)
    idx = np.array(idx).flatten() if len(idx) else np.zeros(0, int)
    persons, ppe = [], []
    for i in idx:
        nm = NAMES[cl[i]]; box = tuple(int(v) for v in xy[i])
        if nm == "person":
            persons.append((box, float(cf[i])))
        else:
            grp = CLASS_TO_GROUP.get(nm)
            if grp in GROUPS and float(cf[i]) >= GROUP_THRESH.get(grp, 0.4):
                ppe.append({"label": nm, "bbox": box, "conf": float(cf[i])})
    return persons, ppe


def run(arm, videos):
    EPOCH = int(time.time()*1000)
    tot = defaultdict(float); frames_all = 0; events = []; seen = {g: defaultdict(int) for g in GROUPS}
    ntracks = npersons = 0
    for vid in videos:
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened(): continue
        stride = max(1, int(round((cap.get(cv2.CAP_PROP_FPS) or 25) / a.target_fps)))
        tracker = BYTETracker(targs); ST = {}; frames = raw = 0
        while frames < a.max_frames:
            ok, frame = cap.read()
            if not ok: break
            raw += 1
            if (raw-1) % stride: continue
            frames += 1; frames_all += 1; H, W = frame.shape[:2]

            t = time.perf_counter()
            if arm == "ours":
                boxes, kpts = ours.people(frame, {}, "x")
                t_det = time.perf_counter()-t
                t = time.perf_counter(); pboxes, verdicts = ours.perceive(frame, boxes, kpts)
                t_cls = time.perf_counter()-t
                dets = np.array([[*map(float, b), 0.9] for b in pboxes], np.float32) if pboxes else np.zeros((0, 5), np.float32)
            else:
                persons, ppe = prod_detect(frame)
                t_det = time.perf_counter()-t; t_cls = 0.0
                pboxes = [p[0] for p in persons]
                dets = np.array([[*map(float, p[0]), p[1]] for p in persons], np.float32) if persons else np.zeros((0, 5), np.float32)
            npersons += len(pboxes)
            tot["det"] += t_det; tot["cls"] += t_cls

            t = time.perf_counter()
            try: tracks = tracker.update(dets, (H, W), (H, W))
            except Exception: tracks = []
            tot["trk"] += time.perf_counter()-t

            active = [{"id": int(getattr(tr, "track_id", -1)), "bbox": tuple(int(v) for v in tr.tlbr),
                       "apd": {g: "unknown" for g in GROUPS}} for tr in tracks if getattr(tr, "tlbr", None) is not None]
            t = time.perf_counter()
            if arm == "ours": assign_ours(active, pboxes, verdicts)
            else: assign_apd_smart(active, ppe, CLASS_TO_GROUP, APD_GROUPS, GROUP_THRESH)

            for tr in active:
                st = ST.setdefault(tr["id"], {"first": frames, "last_ms": 0,
                                              "h": {g: deque(maxlen=VOTE_WINDOW) for g in GROUPS}})
                for g in GROUPS:
                    cur = tr["apd"].get(g, "unknown").lower()
                    cur = cur if cur in ("ok", "violation") else "unknown"
                    st["h"][g].append(cur); seen[g][cur] += 1
            for tr in active:
                st = ST[tr["id"]]
                if frames - st["first"] < MIN_AGE: continue
                voted = [g for g in GROUPS if len(st["h"][g]) and
                         sum(1 for v in st["h"][g] if v == "violation")/len(st["h"][g]) >= RATIO]
                if not voted or st.get("state") == tuple(sorted(voted)): continue
                now = EPOCH + int(frames/a.target_fps*1000)
                if now - st["last_ms"] < COOLDOWN: continue
                st["last_ms"] = now; st["state"] = tuple(sorted(voted))
                events.append({"v": Path(vid).name, "f": frames, "t": tr["id"], "g": voted})
            tot["vote"] += time.perf_counter()-t
        ntracks += len(ST); cap.release()
    return dict(frames=frames_all, persons=npersons, tracks=ntracks, events=events, seen=seen, t=tot)


res = {arm: run(arm, a.videos) for arm in ("prod", "ours")}
secs = res["ours"]["frames"]/a.target_fps

print(f"{'':6} {'ppl/f':>6} {'tracks':>7} {'detect':>8} {'classify':>9} {'track':>7} {'total':>8} {'events':>7} {'/hour':>8}")
for arm, r in res.items():
    f = max(1, r["frames"]); per = sum(r["t"].values())/f*1000
    print(f"{arm:6} {r['persons']/f:6.2f} {r['tracks']:7d} {r['t']['det']/f*1000:7.1f}m {r['t']['cls']/f*1000:8.1f}m "
          f"{r['t']['trk']/f*1000:6.1f}m {per:7.1f}m {len(r['events']):7d} {len(r['events'])/secs*3600:8.0f}")

print(f"\nper-track-frame verdicts     {'ok':>7} {'violation':>10} {'unknown':>8}   silence")
for arm, r in res.items():
    for g in GROUPS:
        s = r["seen"][g]; tot_g = sum(s.values()) or 1
        print(f"  {arm:5} {g:9}          {s['ok']:7d} {s['violation']:10d} {s['unknown']:8d}   {s['unknown']/tot_g:6.1%}")

print("\nevents per group")
for arm, r in res.items():
    c = defaultdict(int)
    for e in r["events"]:
        for g in e["g"]: c[g] += 1
    print(f"  {arm:5} " + "  ".join(f"{g} {c[g]}" for g in GROUPS))
