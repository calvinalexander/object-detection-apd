"""Virtual tray-counter run.

A synthetic 2-minute serving-counter clip (1280x720 @ 15 fps, same as the recorder)
is fed through the PRODUCTION tracker + line counter (tlabcode/tracker.py,
tlabcode/line_counter.py) with the PRODUCTION .env thresholds. The YOLO model is
replaced by a detector stand-in that returns the ground-truth boxes with
controllable errors (missed frames, fill-level confusion, box jitter, false boxes),
because the compiled .axnet only runs on a Metis card.

    python sim.py --runs 30            # stats for every detector profile
    python sim.py --render realistic   # also write the annotated demo video
"""
import argparse, json, os, sys, random, glob, logging
from collections import defaultdict
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TC = "/home/ubuntu/tryan/apd/artifacts/tray_counter"
os.environ["PROJECT_DIR"] = os.path.join(HERE, "run")
os.environ["RECORDINGS_PATH"] = os.path.join(HERE, "run", "recordings")
os.environ["VIOLATION_DIR"] = os.path.join(HERE, "run", "violations")
sys.path.insert(0, TC)
from tlabcode.config import load_config_from_env, LOG            # noqa: E402
from tlabcode.tracker import AdvancedObjectTracker                # noqa: E402
from tlabcode import line_counter as LC                           # noqa: E402
from tlabcode.visualizer import draw_counting_line, draw_detections, draw_statistics, draw_violation_highlight  # noqa: E402

LOG.setLevel(logging.WARNING)
for h in LOG.handlers: h.setLevel(logging.WARNING)

W, H, FPS, SECS = 1280, 720, 15, 120
N = FPS * SECS
LINE = [(640, 650), (700, 120)]          # virtual camera's ROI; normal points right = OUT
TW, TH = 170, 115                         # tray size in pixels (area ~19.5k >> MIN_TRAY_AREA)
NAMES = {0: "empty", 1: "full", 2: "half", 3: "<half", 4: ">half", 5: "stack"}
VALID_OUT = {1, 5}
FILL_ORDER = [0, 3, 2, 4, 1]              # empty < less-than-half < half < more-than-half < full

# ------------------------------------------------------------------ scenario
def build_scenario():
    """Each tray: path of (frame -> centre) and a class. Returns list of dicts."""
    T = []
    def tray(t0, lane_y, cls, speed=11.0, direction="out", stops=None, note=""):
        T.append(dict(t0=int(t0 * FPS), y=lane_y, cls=cls, v=speed, dir=direction,
                      stops=stops or [], note=note))
    lanes = [230, 350, 470]
    # 0-40 s: steady line of full trays, two lanes
    for i, t in enumerate(np.arange(1, 40, 2.6)):
        tray(t, lanes[i % 2], 1, speed=random.uniform(9, 13))
    # underfilled trays mixed into the flow -> should be violations
    tray(6.0, lanes[2], 3, note="less-than-half going out")
    tray(14.5, lanes[2], 0, note="empty tray going out")
    tray(22.0, lanes[2], 2, note="half-filled going out")
    tray(31.0, lanes[2], 4, note="more-than-half going out")
    # 42-70 s: busier, three lanes, a few stacks
    for i, t in enumerate(np.arange(42, 70, 1.9)):
        tray(t, lanes[i % 3], 5 if i in (3, 9) else 1, speed=random.uniform(10, 14))
    tray(42.0, 590, 3, note="less-than-half going out")
    tray(74.0, 590, 2, note="half-filled going out")
    # returned trays (IN) while the flow is still going out
    tray(52.0, 590, 1, direction="in", note="tray returned against the flow (busy)")
    tray(64.0, 590, 4, direction="in", note="tray returned against the flow (busy)")
    # 72-80 s: tray stops ON the line and wobbles, then goes out
    tray(73.0, 350, 1, speed=10, stops=[(0.5, 2.5)], note="stops on the line and wobbles")
    # 84-90 s: two trays pushed nose-to-tail in the same lane
    tray(84.0, 350, 1, speed=12, note="nose-to-tail pair (front)")
    tray(84.0 + (TW + 12) / 12 / FPS, 350, 1, speed=12, note="nose-to-tail pair (back)")
    # 95 s: tray goes out then is pulled back in 4 s later
    tray(95.0, 230, 0, speed=12, direction="outback", note="empty tray out, then pulled back in")
    # 108 s: a lone tray returned while nothing else moves
    tray(106.0, 470, 1, speed=11, direction="in", note="tray returned, nothing else on screen")
    # 114 s: last few trays out
    tray(113.0, 230, 1, speed=14); tray(114.0, 350, 5, speed=14, note="stack going out")
    for k, t in enumerate(T): t["id"] = k
    for f in range(N):                               # physically possible: no two trays overlap
        bs = [(t["id"], gt_box(t, f)) for t in T]
        bs = [(i, b) for i, b in bs if b is not None]
        for x in range(len(bs)):
            for y in range(x + 1, len(bs)):
                assert iou(bs[x][1], bs[y][1]) == 0, f"trays {bs[x][0]} and {bs[y][0]} overlap at frame {f}"
    return T

def xline(y):                                   # x of the counting line at image row y
    (x1, y1), (x2, y2) = LINE
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)

def centre(t, f):
    """Centre of tray t at frame f, or None if not in the scene."""
    s = (f - t["t0"])
    if s < 0: return None
    y = t["y"]; v = t["v"]; xl = xline(y + TH / 2)
    if t["dir"] == "in":
        x = W + TW - v * s
    elif t["dir"] == "outback":                      # out, park 2s beyond line, come back
        go = (xl + 160 + TW) / v                      # frames to reach park spot
        if s < go: x = -TW + v * s
        elif s < go + 2 * FPS: x = -TW + v * go
        else: x = -TW + v * go - v * (s - go - 2 * FPS)
        if x < -TW - 5 and s > go: return None
    else:
        x = -TW + v * s
        for a, b in t["stops"]:                       # stop at line: (delay into line, duration) seconds
            reach = (xl + TW) / v                     # frames until centre at line
            if s > reach:
                hold = b * FPS
                ds = min(s - reach, hold)
                x = -TW + v * reach + 14 * np.sin(ds / 3.0)   # wobble +-14 px on the line
                if s - reach > hold: x = -TW + v * reach + v * (s - reach - hold)
    if x > W + TW + 5 or x < -TW - 5: return None
    return (x, y)

def gt_box(t, f):
    c = centre(t, f)
    if c is None: return None
    x, y = c
    return (x - TW / 2, y - TH / 2, x + TW / 2, y + TH / 2)

def side(pt):
    (x1, y1), (x2, y2) = LINE
    return np.sign((x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1))

def gt_events(T):
    """Intended ground truth: out-trays = one OUT, returned trays = one IN, out-and-back = OUT+IN.
    Frame = first physical crossing (a tray wobbling on the line is still ONE tray out)."""
    phys = physical_crossings(T)
    want = {"out": ["out"], "in": ["in"], "outback": ["out", "in"]}
    return {t["id"]: [(d, next((f for dd, f in phys[t["id"]] if dd == d), -1)) for d in want[t["dir"]]] for t in T}

def physical_crossings(T):
    """Every time a tray's bottom-centre changes side of the line."""
    ev = defaultdict(list)
    for t in T:
        prev = None
        for f in range(N):
            b = gt_box(t, f)
            if b is None: continue
            s = side(((b[0] + b[2]) / 2, b[3]))
            if prev is not None and s != prev and s != 0:
                ev[t["id"]].append(("out" if s > 0 else "in", f))   # left->right = out (line normal points right)
            if s != 0: prev = s
    return ev

# ------------------------------------------------------------------ detector stand-in
PROFILES = {
    "ideal":     dict(miss=0.00, flip=0.00, jit=0.0, fp=0.00),
    "realistic": dict(miss=0.05, flip=0.03, jit=3.0, fp=0.01),
    "harsh":     dict(miss=0.15, flip=0.10, jit=6.0, fp=0.03),
}

def neighbour(c, rng):
    if c == 5: return 1
    i = FILL_ORDER.index(c)
    j = min(len(FILL_ORDER) - 1, max(0, i + rng.choice([-1, 1])))
    return FILL_ORDER[j]

class Detector:
    def __init__(self, prof, rng):
        self.p, self.rng, self.missing = prof, rng, set()
    def __call__(self, T, f):
        p, rng, out = self.p, self.rng, []
        for t in T:
            b = gt_box(t, f)
            if b is None: continue
            vis = (min(b[2], W) - max(b[0], 0)) * (min(b[3], H) - max(b[1], 0)) / (TW * TH)
            if vis < 0.6: continue                              # half out of frame: not detected
            pm = 0.5 if t["id"] in self.missing else p["miss"]  # misses come in short bursts
            if rng.random() < pm: self.missing.add(t["id"]); continue
            self.missing.discard(t["id"])
            j = rng.normal(0, p["jit"], 4) if p["jit"] else np.zeros(4)
            box = [float(np.clip(b[0] + j[0], 0, W)), float(np.clip(b[1] + j[1], 0, H)),
                   float(np.clip(b[2] + j[2], 0, W)), float(np.clip(b[3] + j[3], 0, H))]
            cls = neighbour(t["cls"], rng) if rng.random() < p["flip"] else t["cls"]
            out.append([*box, float(rng.uniform(0.35, 0.95)), cls])
        if rng.random() < p["fp"]:                              # a bowl / hand detected as a tray
            x, y = rng.uniform(0, W - 120), rng.uniform(100, H - 100)
            out.append([x, y, x + rng.uniform(60, 160), y + rng.uniform(50, 110),
                        float(rng.uniform(0.25, 0.5)), int(rng.integers(0, 6))])
        return out

# ------------------------------------------------------------------ sim-only patched variants
def _patched_module(path, name, subs):
    import types
    src = open(path).read()
    for a, b in subs:
        assert a in src, a
        src = src.replace(a, b)
    m = types.ModuleType(name); m.__file__ = path
    exec(compile(src, path, "exec"), m.__dict__)
    return m

PATCHED_LC = _patched_module(f"{TC}/tlabcode/line_counter.py", "line_counter_patched", [
    # direction = which side of the ROI line the tray moved FROM -> TO (recent_sides is already
    # computed by the production code, just not used), instead of the on-screen average flow
    ("                        # ✅ COUNT IT!",
     "                        direction = 'out' if recent_sides[0] > 0 else 'in'\n"
     "                        determination_method = 'SIDE_CHANGE'\n"
     "                        # ✅ COUNT IT!"),
])
PATCHED_TR = _patched_module(f"{TC}/tlabcode/tracker.py", "tracker_patched", [
    # match across classes (a fill-level flicker is not a new tray) ...
    ("if obj_classes[i] == input_classes[j]:", "if True:"),
    # ... and settle the tray's class by majority vote over its detections
    ('self.objects[oid]["bbox"] = input_boxes[j]',
     'self.objects[oid]["bbox"] = input_boxes[j]\n'
     '                    v = self.objects[oid].setdefault("votes", {})\n'
     '                    v[input_classes[j]] = v.get(input_classes[j], 0) + 1\n'
     '                    self.objects[oid]["cls"] = max(v, key=v.get)'),
])

# ------------------------------------------------------------------ one run
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0

def run(T, prof, seed, render=None, sprites=None, variant="production"):
    rng = np.random.default_rng(seed)
    cfg = load_config_from_env()
    cfg.COUNTING_LINE = LINE
    LC.FRAME_BUFFER.clear()
    lc = PATCHED_LC if variant == "patched" else LC
    Tr = PATCHED_TR.AdvancedObjectTracker if variant == "patched" else AdvancedObjectTracker
    lc.FRAME_BUFFER.clear()
    tracker = Tr(track_history_len=cfg.TRACK_HISTORY_LEN)
    counter = lc.SmartLineCrossingCounter(cfg.COUNTING_LINE, None, 1, cfg)
    det = Detector(PROFILES[prof], rng)
    events = []                                  # (frame, dir, counted_cls, gt_tray or None)
    vw = None
    if render:
        vw = cv2.VideoWriter(render, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        bg = make_background()
    for f in range(N):
        dets = det(T, f)
        dets = [d for d in dets if d[4] >= cfg.CONF_THRESHOLD]      # counter.py:415
        tracked = tracker.update(dets, cfg, f + 1)
        before = set(counter.counted_objects)
        frame = None
        if render:
            frame = bg.copy()
            for t in T:
                b = gt_box(t, f)
                if b is not None: paste(frame, sprites[t["id"]], b)
        fe = counter.update(tracked, tracker.trajectories, cfg, f + 1, frame, FPS)
        new = [o for o in counter.counted_objects if o not in before]
        for oid, e in zip(new, fe):
            ob = tracked[oid]["bbox"]
            best, bi = None, 0.2
            for t in T:
                g = gt_box(t, f)
                if g is not None and iou(ob, g) > bi: best, bi = t["id"], iou(ob, g)
            events.append((f, e["direction"], e["cls"], best))
        if render:
            ic, oc = counter.get_counts()
            draw_counting_line(frame, cfg.COUNTING_LINE, cfg)
            draw_detections(frame, dets, tracked, tracker.trajectories, cfg.CLASS_NAMES, cfg.CLASS_COLORS, cfg)
            draw_statistics(frame, cfg.CLASS_NAMES, cfg.CLASS_COLORS, ic, oc, f + 1, N, len(tracked), cfg)
            recent = [e for e in events if f - e[0] < 20 and e[1] == "out" and e[2] not in VALID_OUT]
            for e in recent:
                oid = [o for o, d in counter.counted_objects.items() if d["frame"] == e[0] + 1]
                if oid: draw_violation_highlight(frame, oid[0], tracked, cfg.CLASS_COLORS, cfg)
            tag = "PATCHED" if variant == "patched" else "PRODUCTION"
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
            cv2.rectangle(frame, (W - tw - 40, 12), (W - 12, 58), (40, 40, 40) if variant != "patched" else (120, 90, 10), -1)
            cv2.putText(frame, tag, (W - tw - 26, 47), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"t={f / FPS:5.1f}s  [virtual scene, detector: {prof}]", (W - 470, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(frame)
    counter.close()
    if vw: vw.release()
    return events

# ------------------------------------------------------------------ scoring
def score(T, gt, events):
    by_tray = defaultdict(list)
    phantom = []
    for e in events:
        (by_tray[e[3]] if e[3] is not None else phantom).append(e)
    r = dict(gt_out=0, gt_in=0, sys_out=0, sys_in=0, correct=0, missed=0, extra=0, wrong_dir=0,
             viol_gt=0, viol_tp=0, viol_fp=0, viol_fn=0, phantom=len(phantom))
    per = []
    for t in T:
        exp = [d for d, _ in gt[t["id"]]]
        got = [e[1] for e in by_tray[t["id"]]]
        r["gt_out"] += exp.count("out"); r["gt_in"] += exp.count("in")
        rem = list(got)
        ok = 0
        for d in exp:
            if d in rem: rem.remove(d); ok += 1
        wrong = min(len(exp) - ok, len(rem))
        r["correct"] += ok; r["wrong_dir"] += wrong
        r["missed"] += len(exp) - ok - wrong; r["extra"] += len(rem) - wrong
        gv = "out" in exp and t["cls"] not in VALID_OUT
        pv = any(e[1] == "out" and e[2] not in VALID_OUT for e in by_tray[t["id"]])
        r["viol_gt"] += gv; r["viol_tp"] += gv and pv; r["viol_fp"] += pv and not gv; r["viol_fn"] += gv and not pv
        per.append(dict(id=t["id"], cls=NAMES[t["cls"]], note=t["note"], expected=exp, got=got,
                        got_cls=[NAMES[e[2]] for e in by_tray[t["id"]]],
                        ok=(sorted(exp) == sorted(got)), viol_gt=bool(gv), viol_pred=bool(pv)))
    r["viol_fp"] += sum(1 for e in phantom if e[1] == "out" and e[2] not in VALID_OUT)
    r["sys_out"] = sum(1 for e in events if e[1] == "out"); r["sys_in"] = sum(1 for e in events if e[1] == "in")
    return r, per

# ------------------------------------------------------------------ drawing helpers
def load_sprites(T, rng):
    S = os.path.join(HERE, "sprites")
    pick = {0: ["c0_0", "c0_1", "c0_3"], 1: ["c1_0", "c1_1", "c1_4", "c1_5"], 2: ["c2_2", "c2_0"],
            3: ["c3_1", "c3_2", "c3_0"], 4: ["c4_2", "c4_3"], 5: ["c5_0", "c5_4"]}
    return {t["id"]: cv2.resize(cv2.imread(f"{S}/{rng.choice(pick[t['cls']])}.png"), (TW, TH)) for t in T}

def make_background():
    g = np.random.default_rng(0)
    bg = np.full((H, W, 3), (150, 150, 146), np.uint8).astype(np.float32)
    bg += np.linspace(-18, 18, W)[None, :, None]
    bg += g.normal(0, 4, (H, W, 1))
    cv2.rectangle(bg, (0, 150), (W, 670), (178, 180, 182), -1)            # steel serving counter
    for y in (150, 670): cv2.line(bg, (0, y), (W, y), (95, 95, 95), 3)
    bg[150:670] += g.normal(0, 3, (520, W, 1))
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    cv2.putText(bg, "KITCHEN / SERVING", (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    cv2.putText(bg, "PICK-UP / DINERS", (W - 200, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    return bg

def paste(frame, spr, b):
    x1, y1 = int(round(b[0])), int(round(b[1]))
    sx1, sy1 = max(0, -x1), max(0, -y1)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x1 + TW), min(H, y1 + TH)
    if x2c <= x1c or y2c <= y1c: return
    sh = frame[min(H - 1, y1c + 6):min(H, y2c + 6), min(W - 1, x1c + 6):min(W, x2c + 6)]
    sh[:] = (sh * 0.7).astype(np.uint8)                                   # drop shadow
    frame[y1c:y2c, x1c:x2c] = spr[sy1:sy1 + (y2c - y1c), sx1:sx1 + (x2c - x1c)]

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--render", default=None, help="profile to render as video")
    ap.add_argument("--variant", default="production", choices=["production", "patched"])
    a = ap.parse_args()
    random.seed(7)
    T = build_scenario()
    gt = gt_events(T)
    cfg = load_config_from_env()
    print("production thresholds:", {k: getattr(cfg, k) for k in (
        "CONF_THRESHOLD", "MIN_TRAY_AREA", "IOU_THRESHOLD", "MAX_DISTANCE", "MAX_POSITION_JUMP",
        "MIN_TRAJECTORY_POINTS", "CROSSING_BUFFER_ZONE", "ANTI_BOUNCE_COOLDOWN", "TRACK_HISTORY_LEN")})
    print(f"scenario: {len(T)} trays, GT events: out={sum(d == 'out' for v in gt.values() for d, _ in v)} "
          f"in={sum(d == 'in' for v in gt.values() for d, _ in v)}")
    res = {"trays": len(T), "scenario": [{k: t[k] for k in ("id", "t0", "y", "cls", "dir", "note")} for t in T],
           "profiles": {}}
    for variant in (("production", "patched") if a.runs else ()):
      for prof in PROFILES:
        runs = []
        key = f"{variant}/{prof}"
        for s in range(a.runs):
            ev = run(T, prof, s, variant=variant)
            r, per = score(T, gt, ev)
            runs.append(r)
            if s == 0: res["profiles"].setdefault(key, {})["per_tray_seed0"] = per
            res["profiles"].setdefault(key, {}).setdefault("fail_count", {})
            for p in per:
                if not p["ok"] or p["viol_gt"] != p["viol_pred"]:
                    fc = res["profiles"][key]["fail_count"]; fc[p["id"]] = fc.get(p["id"], 0) + 1
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        agg_min = {k: int(np.min([r[k] for r in runs])) for k in runs[0]}
        agg_max = {k: int(np.max([r[k] for r in runs])) for k in runs[0]}
        res["profiles"][key].update(mean=agg, min=agg_min, max=agg_max, runs=a.runs)
        print(f"\n[{key}] mean over {a.runs} runs")
        print("  " + "  ".join(f"{k}={v:.1f}" for k, v in agg.items()))
    # which scenario trays fail most often
    json.dump(res, open(os.path.join(HERE, f"results_{a.runs}runs.json"), "w"), indent=1, default=str)
    if a.render:
        sprites = load_sprites(T, np.random.default_rng(1))
        ev = run(T, a.render, 0, render=os.path.join(HERE, f"demo_{a.render}_{a.variant}_raw.mp4"),
                 sprites=sprites, variant=a.variant)
        r, per = score(T, gt, ev)
        json.dump(dict(score=r, per=per, events=[list(map(str, e)) for e in ev]),
                  open(os.path.join(HERE, f"demo_{a.render}_{a.variant}.json"), "w"), indent=1)
        print("\nrendered run:", r)
