"""APD runner: pose on the accelerator, ROIs and classification in the app.

The stock ROI transforms cannot place our crops -- `roicrop` takes a fixed
rectangle or a metadata box, and `facealign` is keypoint-driven but
face-specific. Neither can express "1.9x shoulder width, biased 0.65 upward".
So the pipeline splits the way the production service already splits: the
accelerator runs pose, and the application derives ROIs from the keypoints and
classifies them.

Two interchangeable backends, because the logic must be testable off-device:
  axelera      create_inference_stream() over the compiled pose network
  ultralytics  YOLO pose, for development on a workstation

Classification is ONNX Runtime over the exported trunks, which runs on CPU
without compilation. Swapping in AxInferenceNet later changes one class.

Events follow the production contract -- min age, majority over a vote window,
cooldown -- so results are comparable, but they are written locally only. There
is no upload: this must not put rows anywhere the dashboard reads.
"""
import argparse, collections, json, sqlite3, sys, time, uuid
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.pose import build_rois, roi_usable
from apd.schema import GEOMETRY, Attribute, assert_contract

assert_contract()
ATTRS = [a.value for a in Attribute]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="video file or rtsp url")
ap.add_argument("--backend", choices=["ultralytics", "axelera"], default="ultralytics")
ap.add_argument("--pose-weights", default=".work/yolo11n-pose.pt")
ap.add_argument("--pose-network", default="apd_ours", help="axelera network name")
ap.add_argument("--head-onnx", default="runs/packed/apd_ours_head.onnx")
ap.add_argument("--hand-onnx", default="runs/packed/apd_ours_hand.onnx")
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.40)
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--vote-window", type=int, default=5)
ap.add_argument("--min-age-frames", type=int, default=3)
ap.add_argument("--min-violation-ratio", type=float, default=0.60)
ap.add_argument("--cooldown-ms", type=int, default=20000)
ap.add_argument("--gate", type=int, default=1, help="skip ROIs that cannot be read")
ap.add_argument("--outdir", default="apd_local")
ap.add_argument("--save-raw", type=int, default=1,
                help="also write the un-annotated frame; the production service writes "
                     "only the drawn copy, which is why every dashboard image carries "
                     "a burned-in verdict")
a = ap.parse_args()


# ----------------------------------------------------------------- pose
class PoseUltralytics:
    def __init__(self):
        from ultralytics import YOLO
        self.m = YOLO(a.pose_weights)
    def __call__(self, frame):
        r = self.m.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False, device=0)[0]
        if r.keypoints is None or not len(r.boxes):
            return [], []
        return r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()


class PoseAxelera:
    """Frames and keypoints from the compiled network, via InferenceStream."""
    def __init__(self):
        from axelera.app import config as ax_config
        from axelera.app.stream import create_inference_stream
        self.stream = create_inference_stream(
            network=a.pose_network, sources=[a.source],
            hardware_caps=ax_config.HardwareCaps(
                vaapi=ax_config.HardwareEnable.detect,
                opencl=ax_config.HardwareEnable.detect,
                opengl=ax_config.HardwareEnable.detect),
            aipu_cores=1, specified_frame_rate=10)
    def frames(self):
        for f in self.stream:
            meta = f.meta
            boxes = getattr(meta, "boxes", np.zeros((0, 4)))
            kpts = getattr(meta, "keypoints", np.zeros((0, 17, 3)))
            yield f.image.asarray(), np.asarray(boxes), np.asarray(kpts)


# ----------------------------------------------------- classification
class OnnxHeads:
    """head crop -> (head_cap, mask); hand crop -> gloves. CPU, no compilation."""
    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions(); opts.log_severity_level = 3
        self.head = ort.InferenceSession(a.head_onnx, opts, providers=["CPUExecutionProvider"])
        self.hand = ort.InferenceSession(a.hand_onnx, opts, providers=["CPUExecutionProvider"])
    @staticmethod
    def _prep(c):
        x = (cv2.cvtColor(cv2.resize(c, (128, 128)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
        return x.transpose(2, 0, 1)[None]
    def head_attrs(self, crop):
        o = self.head.run(None, {"input": self._prep(crop)})[0][0]
        return float(o[0]), float(o[1])
    def gloves(self, crop):
        return float(self.hand.run(None, {"input": self._prep(crop)})[0][0][0])


def crop_np(img, box, size):
    x1, y1, x2, y2 = box
    pad = 0.35 * max(x2 - x1, y2 - y1)
    c = img[max(0, int(y1-pad)):int(y2+pad), max(0, int(x1-pad)):int(x2+pad)]
    return None if c.size == 0 or min(c.shape[:2]) < 8 else c


# ----------------------------------------------------------- tracking
def make_tracker():
    """Production's vendored ByteTrack if present, else Ultralytics'."""
    for p in ("/data/adin/sppg-ai/src", "/mnt/data/voyager-sdk"):
        if Path(p).is_dir():
            sys.path.insert(0, p)
    try:
        import types
        from apd_tracker.byte_tracker import BYTETracker
        args = types.SimpleNamespace(track_thresh=0.30, track_buffer=60, match_thresh=0.8,
                                     aspect_ratio_thresh=1.6, min_box_area=10, mot20=False)
        print("  tracker: apd_tracker ByteTrack (production's)")
        return ("prod", BYTETracker(args))
    except Exception as e:
        print(f"  tracker: production's unavailable ({type(e).__name__}); using IoU fallback")
        return ("iou", None)


class IoUTracker:
    """Minimal IoU tracker, so the runner works where ByteTrack will not import."""
    def __init__(self, iou_thresh=0.3, buffer=30):
        self.next_id, self.tracks, self.iou, self.buf = 1, {}, iou_thresh, buffer
    @staticmethod
    def _iou(a_, b_):
        ix = max(0, min(a_[2], b_[2]) - max(a_[0], b_[0]))
        iy = max(0, min(a_[3], b_[3]) - max(a_[1], b_[1]))
        inter = ix * iy
        u = (a_[2]-a_[0])*(a_[3]-a_[1]) + (b_[2]-b_[0])*(b_[3]-b_[1]) - inter
        return inter / max(1e-6, u)
    def update(self, boxes, frame_id):
        out, used = [], set()
        for tid, t in list(self.tracks.items()):
            best, bi = 0.0, -1
            for i, b in enumerate(boxes):
                if i in used: continue
                v = self._iou(t["box"], b)
                if v > best: best, bi = v, i
            if best >= self.iou:
                self.tracks[tid] = {"box": boxes[bi], "last": frame_id}
                used.add(bi); out.append((tid, boxes[bi]))
            elif frame_id - t["last"] > self.buf:
                del self.tracks[tid]
        for i, b in enumerate(boxes):
            if i in used: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {"box": b, "last": frame_id}
            out.append((tid, b))
        return out


# ---------------------------------------------------------------- store
class Store:
    """Local SQLite + snapshots. Deliberately not the production database."""
    def __init__(self, root):
        self.root = Path(root); (self.root/"snapshots").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root/"apd_local.db")
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("""CREATE TABLE IF NOT EXISTS events(
            event_id TEXT PRIMARY KEY, timestamp TEXT, frame_id INTEGER, track_id INTEGER,
            person_bbox TEXT, violations TEXT, scores TEXT, image_path TEXT, raw_path TEXT)""")
        self.db.commit()
    def save(self, ev):
        self.db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)",
                        (ev["event_id"], ev["timestamp"], ev["frame_id"], ev["track_id"],
                         json.dumps(ev["person_bbox"]), json.dumps(ev["violations"]),
                         json.dumps(ev["scores"]), ev["image_path"], ev.get("raw_path")))
        self.db.commit()


OK, BAD, UNK = (80,200,120), (60,60,235), (150,155,160)
def draw(im, tid, box, verdicts):
    x1,y1,x2,y2 = [int(v) for v in box]
    col = BAD if "absent" in verdicts.values() else (OK if "present" in verdicts.values() else UNK)
    cv2.rectangle(im, (x1,y1), (x2,y2), col, 2)
    y = max(24, y1-52)
    cv2.rectangle(im, (x1,y), (x1+108,y+50), (18,24,28), -1)
    cv2.putText(im, f"id {tid}", (x1+4,y+13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,210,215), 1)
    for i, at in enumerate(ATTRS):
        s = verdicts[at]; mk = {"present":"OK","absent":"X","unclear":"?"}[s]
        c = OK if s=="present" else (BAD if s=="absent" else UNK)
        cv2.putText(im, f"{mk} {at}", (x1+4, y+26+12*i), cv2.FONT_HERSHEY_SIMPLEX, 0.34, c, 1)
    return im


def main():
    pose = PoseUltralytics() if a.backend == "ultralytics" else PoseAxelera()
    clf = OnnxHeads()
    kind, tracker = make_tracker()
    iou_tracker = IoUTracker() if kind == "iou" else None
    store = Store(a.outdir)
    state = {}
    counts = collections.Counter()
    cap = cv2.VideoCapture(a.source) if a.backend == "ultralytics" else None
    fid = kept = 0
    t0 = time.time()
    print(f"  source: {a.source}\n  output: {a.outdir}/  (local only, no upload)\n")

    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok: break
            fid += 1
            if fid % a.stride: continue
            boxes, kpts = pose(frame)
        else:
            try: frame, boxes, kpts = next(pose.frames())
            except StopIteration: break
            fid += 1
        kept += 1
        if a.max_frames and kept > a.max_frames: break
        if len(boxes) == 0: continue

        people = build_rois(np.asarray(boxes), np.asarray(kpts))
        pboxes = [p.person_box for p in people]
        if iou_tracker is not None:
            tracked = iou_tracker.update([p.person_box for p in people], fid)
        else:
            dets = np.array([[*p.person_box, 0.9] for p in people], dtype=np.float32)
            H, W = frame.shape[:2]
            tracked = [(int(getattr(t, "track_id", -1)), tuple(getattr(t, "tlbr")))
                       for t in tracker.update(dets, (H, W), (H, W))]

        for tid, tbox in tracked:
            pp = min(people, key=lambda q: abs(q.person_box[0]-tbox[0]) + abs(q.person_box[1]-tbox[1]))
            v, sc = {}, {}
            head_ok = (pp.head is not None and not pp.head_collision and
                       (not a.gate or roi_usable(pp.head, frame.shape, part="head",
                                                 others=pboxes, self_box=pp.person_box)))
            hc = crop_np(frame, pp.head, GEOMETRY.head_crop_px) if head_ok else None
            if hc is None:
                v["head_cap"] = v["mask"] = "unclear"
            else:
                p_hc, p_mk = clf.head_attrs(hc)
                sc["head_cap"], sc["mask"] = p_hc, p_mk
                v["head_cap"] = "present" if p_hc >= .5 else "absent"
                v["mask"] = "present" if p_mk >= .5 else "absent"
            hands = []
            for h in pp.hands[:2]:
                if a.gate and not roi_usable(h, frame.shape, part="hand",
                                             others=pboxes, self_box=pp.person_box):
                    continue
                c = crop_np(frame, h, GEOMETRY.hand_crop_px)
                if c is not None: hands.append(c)
            if len(hands) < 2:
                v["gloves"] = "unclear"
            else:
                ps = [clf.gloves(c) for c in hands[:2]]
                sc["gloves"] = min(ps)              # both hands required
                v["gloves"] = "present" if min(ps) >= .5 else "absent"
            for at in ATTRS: counts[f"{at}:{v[at]}"] += 1

            st = state.setdefault(tid, {"first": fid, "last_ms": 0, "last_set": None,
                                        "hist": {at: collections.deque(maxlen=a.vote_window) for at in ATTRS}})
            for at in ATTRS: st["hist"][at].append(v[at])
            if fid - st["first"] < a.min_age_frames: continue
            voted = [at for at in ATTRS
                     if st["hist"][at] and
                     sum(1 for x in st["hist"][at] if x == "absent")/len(st["hist"][at]) >= a.min_violation_ratio]
            if not voted: continue
            if st["last_set"] == tuple(sorted(voted)): continue
            now = int(time.time()*1000)
            if now - st["last_ms"] < a.cooldown_ms: continue

            eid = str(uuid.uuid4())
            snap = Path(a.outdir)/"snapshots"/f"{eid}.jpg"
            cv2.imwrite(str(snap), draw(frame.copy(), tid, pp.person_box, v))
            raw = None
            if a.save_raw:
                raw = Path(a.outdir)/"snapshots"/f"{eid}_raw.jpg"
                cv2.imwrite(str(raw), frame)
            store.save({"event_id": eid, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "frame_id": fid, "track_id": tid,
                        "person_bbox": [int(x) for x in pp.person_box],
                        "violations": voted, "scores": sc,
                        "image_path": str(snap), "raw_path": str(raw) if raw else None})
            st["last_set"] = tuple(sorted(voted)); st["last_ms"] = now
            counts["events"] += 1

    dt = time.time() - t0
    print(f"\n  {kept} frames in {dt:.0f}s ({kept/max(dt,1e-9):.1f} fps)   events: {counts['events']}")
    print(f"  {'attribute':10} {'present':>8} {'absent':>8} {'unclear':>8}")
    for at in ATTRS:
        print(f"  {at:10} {counts[f'{at}:present']:8d} {counts[f'{at}:absent']:8d} {counts[f'{at}:unclear']:8d}")
    print(f"\n  sqlite: {a.outdir}/apd_local.db   snapshots: {a.outdir}/snapshots/")


if __name__ == "__main__":
    main()
