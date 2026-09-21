"""Run the deployed configuration end to end, on real recordings, and time it.

Everything the device will run is here except the compiled pose graph: the
same perception module, the production ByteTracker, and the production vote /
cooldown logic, driven by the live .env values. The point is to know, before
the container starts, whether the deployed settings hold their frame rate and
what they would have written to the database.

    pose (yolo11n) -> anatomical ROI -> packed classifiers
      -> BYTETracker -> per-track vote window -> event

Reported per stage, because only one of them is ours to tune.
"""
import argparse, os, sys, time
from collections import deque, defaultdict
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/tray_counter/tray/tray_counter"
sys.path.insert(0, str(ROOT / "deploy/apd_ours"))
sys.path.insert(0, str(LIVE))                       # production apd_tracker

ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="+", required=True)
ap.add_argument("--max-frames", type=int, default=300, help="processed frames per video")
ap.add_argument("--person-conf", type=float, default=None, help="override APD_THRESH_PERSON")
ap.add_argument("--target-fps", type=float, default=10.0, help="stream specified_frame_rate")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
os.environ.setdefault("APD_OURS_THREADS", str(a.threads))


def live_env(path=LIVE / ".env"):
    """Only the APD_* knobs. Credentials in that file are not read."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line and line.startswith("APD_"):
            k, v = line.split("=", 1); out[k.strip()] = v.strip()
    return out


ENV = live_env()
VOTE_WINDOW = int(ENV.get("APD_VOTE_WINDOW", 5))
MIN_AGE_FRAMES = int(ENV.get("APD_MIN_AGE_FRAMES", 10))
MIN_VIOLATION_RATIO = float(ENV.get("APD_MIN_VIOLATION_RATIO", 0.6))
COOLDOWN_MS = int(ENV.get("APD_COOLDOWN_MS", 600000))
PERSON_CONF = a.person_conf if a.person_conf is not None else float(ENV.get("APD_THRESH_PERSON", 0.4))
GROUPS = ["headcap", "gloves", "mask"]

print("deployed settings replicated from the live .env")
for k, v in (("vote window", VOTE_WINDOW), ("min age frames", MIN_AGE_FRAMES),
             ("min violation ratio", MIN_VIOLATION_RATIO), ("cooldown ms", COOLDOWN_MS),
             ("person conf", PERSON_CONF), ("target fps", a.target_fps),
             ("ort threads", a.threads)):
    print(f"  {k:22} {v}")

from types import SimpleNamespace
from apd_perception_ours import OursPerception, assign_ours
from apd_tracker.byte_tracker import BYTETracker

args = SimpleNamespace(backend="ultralytics", pose_weights=str(ROOT / "yolo11n-pose.pt"),
                       person_conf=PERSON_CONF, gate=1, imgsz=640,
                       track_thresh=0.30, track_buffer=60, match_thresh=0.8,
                       aspect_ratio_thresh=1.6, min_box_area=10, mot20=False)
perception = OursPerception(args)
print(f"  classifier backend     {'PACKED (one call per frame)' if perception.packed is not None else 'two files'}\n")

EPOCH_MS = int(time.time() * 1000)
TOT = defaultdict(float); N = defaultdict(int)
all_events, per_video = [], []

for vid in a.videos:
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print(f"  cannot open {vid}"); continue
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(src_fps / a.target_fps)))       # what specified_frame_rate=10 does
    tracker = BYTETracker(args)
    TRACK_STATE, events = {}, []
    t_pose = t_cls = t_trk = t_vote = 0.0
    frames = raw = npersons = 0
    seen = {g: defaultdict(int) for g in GROUPS}

    while frames < a.max_frames:
        ok, frame = cap.read()
        if not ok: break
        raw += 1
        if (raw - 1) % stride: continue
        frames += 1
        H, W = frame.shape[:2]

        t = time.perf_counter(); boxes, kpts = perception.people(frame, {}, "apd_ours"); t_pose += time.perf_counter() - t
        t = time.perf_counter(); person_boxes, verdicts = perception.perceive(frame, boxes, kpts); t_cls += time.perf_counter() - t
        npersons += len(person_boxes)

        dets = np.array([[float(b[0]), float(b[1]), float(b[2]), float(b[3]), 0.9] for b in person_boxes],
                        dtype=np.float32) if person_boxes else np.zeros((0, 5), np.float32)
        t = time.perf_counter()
        try: tracks = tracker.update(dets, (H, W), (H, W))
        except Exception as e: print("tracker:", e); tracks = []
        t_trk += time.perf_counter() - t

        active = [{"id": int(getattr(tr, "track_id", -1)),
                   "bbox": tuple(int(x) for x in tr.tlbr),
                   "apd": {g: "unknown" for g in GROUPS}} for tr in tracks if getattr(tr, "tlbr", None) is not None]

        t = time.perf_counter()
        assign_ours(active, person_boxes, verdicts)
        for tr in active:
            st = TRACK_STATE.setdefault(tr["id"], {"first_seen_frame": frames, "last_event_ms": 0,
                                                   "apd_history": {g: deque(maxlen=VOTE_WINDOW) for g in GROUPS}})
            for g in GROUPS:
                cur = tr["apd"].get(g, "unknown").lower()
                cur = cur if cur in ("ok", "violation") else "unknown"
                st["apd_history"][g].append(cur); seen[g][cur] += 1

        for tr in active:                                      # production's event rule, verbatim
            st = TRACK_STATE[tr["id"]]
            if frames - st["first_seen_frame"] < MIN_AGE_FRAMES: continue
            voted = []
            for g in GROUPS:
                hist = st["apd_history"][g]
                if not len(hist): continue
                if sum(1 for v in hist if v == "violation") / float(len(hist)) >= MIN_VIOLATION_RATIO:
                    voted.append(g)
            if not voted: continue
            if st.get("last_event_apd_state") == tuple(sorted(voted)): continue
            # Production stamps events with time.time()*1000, so a track's
            # first event always clears the cooldown (last_event_ms is 0).
            # Video time must start from the same epoch or it blocks itself.
            now_ms = EPOCH_MS + int(frames / a.target_fps * 1000)
            if now_ms - st["last_event_ms"] < COOLDOWN_MS: continue
            st["last_event_ms"] = now_ms; st["last_event_apd_state"] = tuple(sorted(voted))
            events.append({"video": Path(vid).name, "frame": frames, "track": tr["id"],
                           "age": frames - st["first_seen_frame"], "violation": voted})
        t_vote += time.perf_counter() - t

    cap.release()
    if not frames: continue
    per_frame = (t_pose + t_cls + t_trk + t_vote) / frames * 1000
    per_video.append({"video": Path(vid).name, "frames": frames, "src_fps": src_fps, "stride": stride,
                      "persons": npersons / frames, "tracks": len(TRACK_STATE), "events": len(events),
                      "pose": t_pose / frames * 1000, "cls": t_cls / frames * 1000,
                      "trk": t_trk / frames * 1000, "vote": t_vote / frames * 1000, "total": per_frame,
                      "seen": seen})
    all_events += events
    for k, v in (("pose", t_pose), ("cls", t_cls), ("trk", t_trk), ("vote", t_vote)):
        TOT[k] += v; N[k] += frames

print(f"{'video':34} {'frm':>4} {'ppl/f':>6} {'pose':>7} {'class':>7} {'track':>7} {'vote':>6} {'total':>7} {'fps':>6} {'trk':>4} {'evt':>4}")
for r in per_video:
    print(f"{r['video'][:34]:34} {r['frames']:4d} {r['persons']:6.2f} {r['pose']:6.1f}m {r['cls']:6.1f}m "
          f"{r['trk']:6.1f}m {r['vote']:5.1f}m {r['total']:6.1f}m {1000/r['total']:6.1f} {r['tracks']:4d} {r['events']:4d}")

if N["pose"]:
    tot = sum(TOT[k] for k in TOT) / N["pose"] * 1000
    print(f"\nper frame: pose {TOT['pose']/N['pose']*1000:.1f} ms, classify {TOT['cls']/N['cls']*1000:.1f} ms, "
          f"track {TOT['trk']/N['trk']*1000:.1f} ms, vote {TOT['vote']/N['vote']*1000:.1f} ms -> {tot:.1f} ms "
          f"({1000/tot:.1f} fps, target {a.target_fps})")
    print(f"budget at {a.target_fps} fps: {1000/a.target_fps:.0f} ms/frame -> "
          f"{'FITS' if tot <= 1000/a.target_fps else 'OVER BUDGET'} with {1000/a.target_fps - tot:+.1f} ms to spare")

agg = {g: defaultdict(int) for g in GROUPS}
for r in per_video:
    for g in GROUPS:
        for k, v in r["seen"][g].items(): agg[g][k] += v
print("\nper-track-frame verdicts (what the vote window sees)")
print(f"{'group':10} {'ok':>8} {'violation':>10} {'unknown':>8}   unknown share")
for g in GROUPS:
    tot_g = sum(agg[g].values()) or 1
    print(f"{g:10} {agg[g]['ok']:8d} {agg[g]['violation']:10d} {agg[g]['unknown']:8d}   {agg[g]['unknown']/tot_g:6.1%}")

print(f"\nevents that would reach the database: {len(all_events)}")
for e in all_events[:20]:
    print(f"  {e['video'][:30]:30} frame {e['frame']:4d} (age {e['age']:3d}) track {e['track']:3d} -> {','.join(e['violation'])}")
