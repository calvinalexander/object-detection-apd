"""Show what tracking changes: identity, stability, and recovered coverage.

Left panel is the pipeline as it stands -- `.predict()` per frame, no identity.
Right panel is `.track()`, where each person keeps an id across frames, so three
things become possible that are impossible on the left:

  aggregation  a verdict is the majority over everything seen of that person so
               far, not one frame's guess. On the demo footage head_cap flips
               between consecutive frames 6.4% of the time and mask 14%, and a
               person does not put a cap on and take it off -- every flip is an
               error the majority removes.
  recovery     a frame where the head is behind a caption or a nearer worker is
               skipped rather than lost: 23% of head observations and 55% of
               hand observations are unusable in their own frame but answerable
               from another frame of the same track.
  provenance   the card shows n observations and how many were skipped, so a
               verdict resting on two glimpses does not look like one resting
               on fifty.

Track colour is stable per id, so an ID SWITCH -- the failure mode that would
make aggregation confidently wrong -- is visible as a colour jump on a person.
"""
import argparse, collections, colorsys, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from apd.pose import build_rois, roi_usable
from apd.schema import GEOMETRY, assert_contract

assert_contract()
ATTRS = ("head_cap", "mask", "gloves")
OK, BAD, UNK = (80, 200, 120), (60, 60, 235), (150, 155, 160)

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", default="artifacts/tracking_demo.mp4")
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.40)
ap.add_argument("--tracker", default="botsort.yaml")
a = ap.parse_args()

pose_a = YOLO(".work/yolo11x-pose.pt")      # stateless
pose_b = YOLO(".work/yolo11x-pose.pt")      # tracked, separate state
y26 = {k: YOLO(f"runs/classify/runs/classify/runs/ci3_{k}/weights/best.pt") for k in ATTRS}
yi = {k: [i for i, v in m.names.items() if v == "present"][0] for k, m in y26.items()}

def colour(tid):
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.72, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)

def crop(img, box, size):
    x1, y1, x2, y2 = box; pad = 0.35 * max(x2 - x1, y2 - y1)
    c = img[max(0, int(y1 - pad)):int(y2 + pad), max(0, int(x1 - pad)):int(x2 + pad)]
    return None if c.size == 0 or min(c.shape[:2]) < 8 else cv2.resize(c, (size, size))

def classify(frame, pp, others):
    """Per-frame verdicts, with the ROI gate deciding what may be asked."""
    out = {}
    ok_head = roi_usable(pp.head, frame.shape, part="head", others=others,
                         self_box=pp.person_box) and not pp.head_collision
    hc = crop(frame, pp.head, GEOMETRY.head_crop_px) if ok_head else None
    for at in ("head_cap", "mask"):
        if hc is None:
            out[at] = None
        else:
            p = float(y26[at].predict(hc, verbose=False, device=0)[0].probs.data[yi[at]])
            out[at] = p
    hs = []
    for h in pp.hands[:2]:
        if not roi_usable(h, frame.shape, part="hand", others=others, self_box=pp.person_box):
            continue
        c_ = crop(frame, h, GEOMETRY.hand_crop_px)
        if c_ is not None:
            hs.append(c_)
    if len(hs) < 2:
        out["gloves"] = None
    else:
        ps = [float(y26["gloves"].predict(c_, verbose=False, device=0)[0].probs.data[yi["gloves"]])
              for c_ in hs]
        out["gloves"] = min(ps)          # both hands required
    return out

def verdict(p):
    return "unclear" if p is None else ("present" if p >= 0.5 else "absent")

CW, CH = 108, 52
def place(free_boxes, x1, y1, x2, y2, W, H):
    def free(x, y):
        if x < 2 or y < 30 or x + CW > W - 2 or y + CH > H - 2:
            return False
        return all(not (x < bx + CW and x + CW > bx and y < by + CH and y + CH > by)
                   for bx, by in free_boxes)
    for cx, cy in ((x1, y1 - CH - 4), (x2 + 4, y1), (x1 - CW - 4, y1), (x1, y2 + 4)):
        if free(cx, cy):
            return cx, cy
    cx, cy = min(max(2, x1), W - CW - 2), 30
    while not free(cx, cy) and cy + CH < H - 2:
        cy += CH + 3
    return cx, cy

def card(im, box, lines, col, placed, extra=""):
    H, W = im.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
    cx, cy = place(placed, x1, y1, x2, y2, W, H)
    placed.append((cx, cy))
    cv2.line(im, (x1 + 2, y1 + 2), (cx + CW // 2, cy + CH // 2), col, 1)
    cv2.rectangle(im, (cx, cy), (cx + CW, cy + CH), (18, 24, 28), -1)
    cv2.rectangle(im, (cx, cy), (cx + CW, cy + CH), col, 1)
    for j, (txt, c2) in enumerate(lines):
        cv2.putText(im, txt, (cx + 5, cy + 13 + 12 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.33, c2, 1)
    if extra:
        cv2.putText(im, extra, (cx + 5, cy + CH - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 180, 190), 1)

def banner(im, title, sub):
    W = im.shape[1]
    cv2.rectangle(im, (0, 0), (W, 28), (18, 24, 28), -1)
    cv2.putText(im, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 242), 1)
    (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(im, sub, (max(8, W - tw - 8), 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 220), 1)

cap = cv2.VideoCapture(a.video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
if a.start:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * FPS))
wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS / a.stride, (W * 2, H))
print(f"{Path(a.video).name}  {W}x{H} -> {W*2}x{H}  {a.out}")

hist = collections.defaultdict(lambda: collections.defaultdict(list))   # tid -> attr -> [p]
skips = collections.defaultdict(collections.Counter)
trail = collections.defaultdict(list)
seen_ids, flips_a, flips_b, prev_a, prev_b = set(), 0, 0, {}, {}
n = written = 0
t0 = time.time()
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if n % a.stride:
        n += 1; continue
    if written >= int(a.seconds * FPS / a.stride):
        break

    # ---------- left: stateless
    left = fr.copy(); placed = []
    ra = pose_a.predict(fr, imgsz=a.imgsz, conf=a.conf, verbose=False, device=0)[0]
    na = 0
    if ra.keypoints is not None and len(ra.boxes):
        people = build_rois(ra.boxes.xyxy.cpu().numpy(), ra.keypoints.data.cpu().numpy())
        pb = [p.person_box for p in people]; na = len(people)
        for k, pp in enumerate(sorted(people, key=lambda p: p.person_box[0])):
            v = classify(fr, pp, pb)
            lines = []
            for at in ATTRS:
                s = verdict(v[at]); mark = "OK" if s == "present" else ("X" if s == "absent" else "?")
                lines.append((f"{mark} {at}", OK if s == "present" else (BAD if s == "absent" else UNK)))
            col = BAD if any(verdict(v[x]) == "absent" for x in ATTRS) else (
                  OK if any(verdict(v[x]) == "present" for x in ATTRS) else UNK)
            card(left, pp.person_box, lines, col, placed)
            key = round(pp.person_box[0] / 40)      # crude identity for flicker counting
            if key in prev_a and prev_a[key] != verdict(v["head_cap"]):
                flips_a += 1
            prev_a[key] = verdict(v["head_cap"])
    banner(left, "no tracking  (current pipeline)", f"persons: {na}   verdict = this frame only")

    # ---------- right: tracked
    right = fr.copy(); placed = []
    rb = pose_b.track(fr, imgsz=a.imgsz, conf=a.conf, persist=True,
                      tracker=a.tracker, verbose=False, device=0)[0]
    nb = 0
    if rb.boxes is not None and rb.boxes.id is not None and rb.keypoints is not None:
        ids = rb.boxes.id.int().cpu().tolist()
        people = build_rois(rb.boxes.xyxy.cpu().numpy(), rb.keypoints.data.cpu().numpy())
        pb = [p.person_box for p in people]; nb = len(people)
        order = sorted(zip(ids, people), key=lambda t: t[1].person_box[0])
        for tid, pp in order:
            seen_ids.add(tid)
            v = classify(fr, pp, pb)
            for at in ATTRS:
                if v[at] is None:
                    skips[tid][at] += 1
                else:
                    hist[tid][at].append(v[at])
            col = colour(tid)
            x1, y1, x2, y2 = [int(t) for t in pp.person_box]
            trail[tid].append(((x1 + x2) // 2, y2))
            for p0, p1 in zip(trail[tid][-25:], trail[tid][-24:]):
                cv2.line(right, p0, p1, col, 2)
            lines = []
            for at in ATTRS:
                ps = hist[tid][at]
                if not ps:
                    s = "unclear"
                else:
                    s = "present" if sum(1 for p in ps if p >= .5) > len(ps) / 2 else "absent"
                mark = "OK" if s == "present" else ("X" if s == "absent" else "?")
                lines.append((f"{mark} {at} {len(ps):d}",
                              OK if s == "present" else (BAD if s == "absent" else UNK)))
                if at == "head_cap":
                    if tid in prev_b and prev_b[tid] != s:
                        flips_b += 1
                    prev_b[tid] = s
            nobs = len(hist[tid]["head_cap"]); nsk = skips[tid]["head_cap"]
            card(right, pp.person_box, lines, col,
                 placed, extra=f"id {tid}  seen {nobs} skip {nsk}")
    banner(right, "tracked  (BoT-SORT, majority per track)",
           f"persons: {nb}   ids so far: {len(seen_ids)}")

    wr.write(np.concatenate([left, right], axis=1))
    written += 1; n += 1
    if written % 25 == 0:
        print(f"  {written} frames  {written/(time.time()-t0):.1f} fps", flush=True)

cap.release(); wr.release()
print(f"\n{written} frames -> {a.out}")
print(f"head_cap verdict changes: no-tracking {flips_a}   tracked (majority) {flips_b}")
print(f"distinct track ids: {len(seen_ids)}")
long = {t: h for t, h in hist.items() if len(h['head_cap']) >= 5}
print(f"\n{'id':>4} {'obs':>4} {'skip':>5}  per-track agreement with its own majority")
for tid, h in sorted(long.items())[:12]:
    ps = h["head_cap"]; v = [p >= .5 for p in ps]; maj = sum(v) > len(v) / 2
    print(f"{tid:>4} {len(ps):4d} {skips[tid]['head_cap']:5d}  head_cap {sum(1 for x in v if x==maj)/len(v):5.0%}"
          f"   (a person's cap does not change: the remainder are errors)")
