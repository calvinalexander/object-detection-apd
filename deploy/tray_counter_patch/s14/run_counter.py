"""Run production vs patched tlabcode on the cached s14 detections; optionally render side by side."""
import os, sys, json, types, logging, argparse
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
TC = "/home/ubuntu/tryan/apd/artifacts/tray_counter"
PATCH = "/home/ubuntu/tryan/apd/deploy/tray_counter_patch/tlabcode"
V = f"{TC}/data/s14_all_cases_elev45_az35_dist1700_hfov55.mp4"
os.environ.update(PROJECT_DIR=f"{HERE}/run", RECORDINGS_PATH=f"{HERE}/run/rec", VIOLATION_DIR=f"{HERE}/run/viol")
sys.path.insert(0, TC)
os.chdir(TC)                                                       # load_dotenv() finds the device .env
from tlabcode.config import load_config_from_env, LOG
from tlabcode import tracker as TR0, line_counter as LC0
from tlabcode.visualizer import draw_counting_line, draw_detections, draw_statistics, draw_violation_highlight
os.chdir(HERE)
LOG.setLevel(logging.WARNING); [h.setLevel(logging.WARNING) for h in LOG.handlers]

def load(path, name):
    m = types.ModuleType(name); m.__file__ = path
    exec(compile(open(path).read(), path, "exec"), m.__dict__); return m
TR1, LC1 = load(f"{PATCH}/tracker.py", "tr_patched"), load(f"{PATCH}/line_counter.py", "lc_patched")

U = np.array([0.918, 0.397]); NRM = np.array([-U[1], U[0]])       # belt axis (downstream) and its normal
D = json.load(open(f"{HERE}/dets15.json")); STEP = 2

def make_line():
    """Across both lanes, rail to rail, along the belt's true cross direction in this perspective view.
    Belt edges meet at V1 = (-1840, -868), on the horizon of a 45-degree-elevation camera (hfov 55, f = 1229 px).
    The cross-belt vanishing point is V2 = (1858, -868), from (V1 - c).(V2 - c) = -f^2; the line runs from the
    lower lane's rail to the upper lane's rail through V2 and the lane divider at x = 640, plus 20 px each end.
    p1 (bottom) -> p2 (top) puts the line normal downstream, so OUT = downstream."""
    return [(485, 524), (794, 210)]

def run(variant, line, render=None):
    tr_mod, lc_mod = (TR1, LC1) if variant == "patched" else (TR0, LC0)
    cfg = load_config_from_env(); cfg.COUNTING_LINE = line
    lc_mod.FRAME_BUFFER.clear()
    tracker = tr_mod.AdvancedObjectTracker(track_history_len=cfg.TRACK_HISTORY_LEN)
    counter = lc_mod.SmartLineCrossingCounter(line, None, 1, cfg)
    cap = cv2.VideoCapture(V) if render else None
    ev, frames = [], []
    for k, dets in enumerate(D):
        dd = [d[:6] for d in dets if d[4] >= cfg.CONF_THRESHOLD]
        tracked = tracker.update(dd, cfg, k + 1)
        img = None
        if render:
            ok, img = cap.read()
            for _ in range(STEP - 1): cap.read()                   # 30 -> 15 fps by default
        before = set(counter.counted_objects)
        fe = counter.update(tracked, tracker.trajectories, cfg, k + 1, img, 15)
        new = [o for o in counter.counted_objects if o not in before]
        for oid, e in zip(new, fe):
            ev.append(dict(frame=k, t=round(k / 15, 2), dir=e["direction"], cls=int(e["cls"]), oid=int(oid),
                           bbox=[round(float(v)) for v in tracked[oid]["bbox"]]))
        if render:
            ic, oc = counter.get_counts()
            draw_counting_line(img, line, cfg)
            draw_detections(img, dd, tracked, tracker.trajectories, cfg.CLASS_NAMES, cfg.CLASS_COLORS, cfg)
            draw_statistics(img, cfg.CLASS_NAMES, cfg.CLASS_COLORS, ic, oc, k + 1, len(D), len(tracked), cfg)
            # the production stats box has no violation tally; add the counter's own count under Tracked
            cv2.putText(img, f"VIOLATIONS: {sum(counter.violation_counts.values())}", (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
            for e in ev:
                if 0 <= k - e["frame"] < 22 and e["dir"] == "out" and e["cls"] not in cfg.VALID_OUT_CLASSES:
                    draw_violation_highlight(img, e["oid"], tracked, cfg.CLASS_COLORS, cfg)
            tag = variant.upper()
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
            cv2.rectangle(img, (1280 - tw - 40, 12), (1268, 58), (120, 90, 10) if variant == "patched" else (40, 40, 40), -1)
            cv2.putText(img, tag, (1280 - tw - 26, 47), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"t={k / 15:6.1f}s  [s14 render, stand-in detector]", (860, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            render.write(img)
    counter.close()
    return ev, cfg

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--render", action="store_true")
    ap.add_argument("--dets", default=f"{HERE}/dets15.json", help="detections JSON (one list per 15 fps frame)")
    ap.add_argument("--video", default=V, help="video the detections came from (for --render)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _x4")
    ap.add_argument("--src-step", type=int, default=2, help="source frames per counted frame when rendering (2 = 30->15 fps)")
    a = ap.parse_args()
    D = json.load(open(a.dets)); V = a.video; STEP = a.src_step
    line = make_line(); print("counting line:", line)
    res = {"line": line}
    for v in ("production", "patched"):
        w = cv2.VideoWriter(f"{HERE}/s14{a.tag}_{v}_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 720)) if a.render else None
        ev, cfg = run(v, line, w)
        if w: w.release()
        res[v] = ev
        n_out = sum(e["dir"] == "out" for e in ev); n_in = len(ev) - n_out
        viol = sum(e["dir"] == "out" and e["cls"] not in cfg.VALID_OUT_CLASSES for e in ev)
        print(f"{v:10s} OUT={n_out:3d} IN={n_in:3d} violations={viol:3d}")
    json.dump(res, open(f"{HERE}/s14{a.tag}_events.json", "w"), indent=1)
