"""Run the production monitor loop on one recording, as the device would.

    python scripts/105_replay_one.py {ours|prod} <video> <outdir>

ours  the shipped deploy/apd_ours monitor with the bundle's perception
prod  production's own monitor from the tray_counter tree, with its own
      ultralytics backend and a production detector (apd_mbg OpenVINO build,
      a stand-in: the device runs apd_v3, whose weights are not here)

Both read frames at 10 per second by timestamp, which is what the device's
stream does (specified_frame_rate=10), stamp events with the time the frame
was filmed (UTC, from the clip name), and merge boxes as the device's decoders
do. Every APD_* value that decides a snapshot matches the device's environment:
vote window 5, min age 10 frames, violation ratio 0.6, cooldown 600000 ms,
person and PPE thresholds 0.7, detector confidence 0.25. Run with cwd=<outdir> so production's settings file is not picked up.
"""
import os, sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT/"artifacts/tray_counter/tray/tray_counter"
BUNDLE = Path(os.environ.get("APD_BUNDLE") or ROOT/"deploy/apd_ours")   # APD_BUNDLE=deploy/apd_person runs the one-network staging build
PROD_MODEL = ROOT/"artifacts/sdk/sdk/build/apd_mbg_openvino_model"
arm, video, outdir = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()

# live settings that decide what gets reported, copied from the device .env
for k, v in {"APD_VOTE_WINDOW": "5", "APD_MIN_AGE_FRAMES": "10", "APD_MIN_VIOLATION_RATIO": "0.6",
             "APD_COOLDOWN_MS": "600000", "APD_THRESH_HEADCAP": "0.7", "APD_THRESH_MASK": "0.7",
             "APD_THRESH_GLOVES": "0.7", "APD_THRESH_PERSON": "0.7",
             "PROJECT_DIR": str(outdir), "APD_STORAGE_PATH": str(outdir/"apd_storage"),
             "APD_BACKEND": "ultralytics", "APD_MODE": "stream", "USE_REMOTE_CONFIG": "false",
             "SHOW_DISPLAY": "False", "SAVE_VIDEO": "False"}.items():
    os.environ[k] = v
# apd_config calls load_dotenv(), which finds the tree's .env with live
# credentials; values already set are never overridden, so set them to dummies
for k in ("API_KEY", "API_BASE_URL", "API_ENDPOINT", "CONFIG_API_URL", "AWS_ACCESS_KEY_ID",
          "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT_URL", "S3_BUCKET", "EDGE_DEVICE_ID", "SPPG_CODE",
          "SPPG_ID", "APD_DEVICE_ID"):
    os.environ[k] = "offline-replay"

import cv2
_Real = cv2.VideoCapture


import calendar, re, time as _time
# Video clock. The recorder names each clip after its start time in UTC (the
# camera's burned-in clock shows the same moment in WIB, +7h), and the device
# stamps events in UTC (TZ=UTC in its environment). Driving the monitor's clock
# from the video means replay events carry the time the frame was filmed, so
# they can be matched to the device's own events, and the 10-minute cooldown
# counts video time as it does on the device.
_m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", video.name)
CLIP_START = calendar.timegm(tuple(int(x) for x in _m.groups()) + (0, 0, 0)) if _m else None
_clock = {"t": CLIP_START}


class TenFps:
    """cv2.VideoCapture that hands out frames at 10 per second of video time."""
    def __init__(self, src, *a):
        self.c = _Real(src, *a)
        self.fps = self.c.get(cv2.CAP_PROP_FPS) or 25.0
        self.step = self.fps / 10.0
        self.i, self.next = 0, 0.0

    def read(self):
        while True:
            if self.i + 1e-9 >= self.next:
                ok, f = self.c.read(); self.i += 1; self.next += self.step
                if ok and CLIP_START is not None:
                    _clock["t"] = CLIP_START + (self.i - 1) / self.fps
                return ok, f
            if not self.c.grab(): return False, None
            self.i += 1

    def __getattr__(self, n): return getattr(self.c, n)


class VideoTime:
    """The monitor's `time` module, with time() and strftime() on the video clock."""
    def time(self):
        return _clock["t"] if _clock["t"] is not None else _time.time()

    def strftime(self, fmt, t=None):
        return _time.strftime(fmt, _time.gmtime(self.time()) if t is None else t)

    def __getattr__(self, n): return getattr(_time, n)


cv2.VideoCapture = TenFps

# Box merging as on the device. The accelerator pipelines merge overlapping boxes
# at IoU 0.45, keep at most 300, and apd_v3's decoder does it across classes
# (nms_class_agnostic: true), so a head_cap and a no_head_cap box on one head
# cannot both survive. Ultralytics defaults are IoU 0.7 and per-class, so they
# are overridden here rather than in production's code.
from ultralytics.engine.model import Model
_predict = Model.predict


def _device_nms(self, *a, **k):
    k.setdefault("iou", 0.45); k.setdefault("max_det", 300)
    if arm == "prod": k.setdefault("agnostic_nms", True)
    return _predict(self, *a, **k)


Model.predict = _device_nms

sys.path[:0] = ([str(BUNDLE)] if arm == "ours" else []) + [str(TREE)]
os.chdir(outdir)
if arm == "ours":
    import importlib.util
    spec = importlib.util.spec_from_file_location("apd_monitor_axelera_multi", BUNDLE/"apd_monitor_axelera_multi.py")
else:
    import importlib.util
    spec = importlib.util.spec_from_file_location("apd_monitor_axelera_multi", TREE/"apd_monitor_axelera_multi.py")
mon = importlib.util.module_from_spec(spec); spec.loader.exec_module(mon)
if CLIP_START is not None:
    mon.time = VideoTime()          # event timestamps and cooldown on video time

args = SimpleNamespace(
    model=str(PROD_MODEL) if arm == "prod" else "apd_ours",
    sources=[str(video)], cam_ids=[video.stem],
    # REPLAY_VIDEO=<path.mp4> also writes the annotated stream the monitor draws
    # (mode "table": the same draw_table caption as the snapshots, every frame)
    mode="table" if os.environ.get("REPLAY_VIDEO") else "text", pipe="torch-aipu",
    backend="ultralytics", device=None, imgsz=640,
    save=bool(os.environ.get("REPLAY_VIDEO")), display=False,
    outdir="apd_storage/snapshots", outpath=os.environ.get("REPLAY_VIDEO") or "out.mp4", fps=10.0, show_fps=False,
    track_thresh=0.30, track_buffer=60, match_thresh=0.8, aspect_ratio_thresh=1.6,
    min_box_area=10, mot20=False, stride=1)
mon.apd_monitor_main(args)
