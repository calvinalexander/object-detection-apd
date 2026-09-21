"""Side-by-side demo: production's detector against our pipeline, on recordings.

Both panes are rendered into one frame and streamed as MJPEG, so they cannot
drift apart and the video is decoded once. Every toggle is a query parameter,
so changing one simply restarts the stream.

    python demo/app.py            then open http://<host>:8000
"""
import io, json, os, sys, threading, time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "deploy/apd_ours"))
from apd.pose import build_rois, roi_usable
from apd.schema import GEOMETRY
from apd_perception_ours import _crop, _stack, _letterbox, DET_NAMES, _facing_camera

VIDEO_DIR = ROOT / "data/recording"
PROD_XML = ROOT / "artifacts/sdk/sdk/build/apd_mbg_openvino_model/best.xml"
PROD_XML_OLD = ROOT / "model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml"
POSE_W = ROOT / "yolo11s-pose.pt"   # matches the device bundle since 2026-09-17
PACKED = ROOT / "deploy/apd_ours/apd_ours_packed.onnx"
GLOVES = ROOT / "deploy/apd_ours/apd_ours_gloves.onnx"
MASK = ROOT / "deploy/apd_ours/apd_ours_mask.onnx"

GREEN, RED, GREY, WHITE = (60, 200, 60), (60, 60, 235), (150, 150, 150), (235, 235, 235)
COL = {"ok": GREEN, "violation": RED, "unknown": GREY}

app = Flask(__name__, static_folder=str(Path(__file__).parent))
_M = {}
_lock = threading.Lock()


def models():
    """Loaded once, on first request, and shared by every stream."""
    with _lock:
        if _M: return _M
        import onnxruntime as ort, openvino as ov
        from ultralytics import YOLO
        so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.log_severity_level = 3
        core = ov.Core(); dev = "GPU" if "GPU" in core.available_devices else "CPU"
        _M["pose"] = YOLO(str(POSE_W))
        _M["packed"] = ort.InferenceSession(str(PACKED), so, providers=["CPUExecutionProvider"])
        _M["gloves"] = ort.InferenceSession(str(GLOVES), so, providers=["CPUExecutionProvider"])
        _M["mask"] = ort.InferenceSession(str(MASK), so, providers=["CPUExecutionProvider"]) if MASK.exists() else None
        xml = PROD_XML if PROD_XML.exists() else PROD_XML_OLD
        _M["prod"] = core.compile_model(core.read_model(str(xml)), dev)
        _M["prod_name"] = xml.parent.name
        print(f"models ready: production {xml.name} on {dev}", flush=True)
        return _M


# ------------------------------------------------------------------ production
def prod_detect(net, im, conf):
    h, w = im.shape[:2]; s = 640
    r = min(s/h, s/w); nh, nw = round(h*r), round(w*r)
    o = np.full((s, s, 3), 114, np.uint8); t, l = (s-nh)//2, (s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (nw, nh))
    out = net([(o.astype(np.float32)/255).transpose(2, 0, 1)[None]])[net.outputs[0]][0]
    if out.shape[0] < out.shape[1]: out = out.T
    sc = out[:, 4:11]; cl = sc.argmax(1); cf = sc.max(1); m = cf >= conf
    b, cf, cl = out[m][:, :4], cf[m], cl[m]
    if not len(b): return []
    xy = np.stack([b[:, 0]-b[:, 2]/2, b[:, 1]-b[:, 3]/2, b[:, 0]+b[:, 2]/2, b[:, 1]+b[:, 3]/2], 1)
    xy[:, [0, 2]] -= l; xy[:, [1, 3]] -= t; xy /= r
    idx = cv2.dnn.NMSBoxes([[float(v[0]), float(v[1]), float(v[2]-v[0]), float(v[3]-v[1])] for v in xy],
                           cf.tolist(), conf, 0.45)
    idx = np.array(idx).flatten() if len(idx) else np.zeros(0, int)
    return [(xy[i], DET_NAMES[cl[i]], float(cf[i])) for i in idx]


# ------------------------------------------------------------------ ours
def ours_detect(M, im, o):
    """o holds the toggles. Returns a list of (person_box, verdicts, rois)."""
    r = M["pose"].predict(im, imgsz=640, conf=o["pose_conf"], verbose=False)[0]
    if not len(r.boxes): return []
    boxes = r.boxes.xyxy.cpu().numpy(); kpts = r.keypoints.data.cpu().numpy()
    people = build_rois(boxes, kpts)
    pboxes = [p.person_box for p in people]
    out = []
    for pi, pp in enumerate(people):
        v, conf, roi = {}, {}, {}
        facing = (not o["facing_gate"]) or _facing_camera(kpts[pi], 0.5, True)
        head_ok = pp.head is not None and not pp.head_collision
        if head_ok and o["gate"]:
            head_ok = bool(roi_usable(pp.head, im.shape, part="head", others=pboxes, self_box=pp.person_box))
        if (o["headcap"] or o["mask"] != "off") and head_ok:
            hc = _crop(im, pp.head, GEOMETRY.head_crop_px)
            if hc is not None:
                p_cap, p_mask, _ = M["packed"].run(None, {"head_crops": _stack([hc]), "hand_crops": _stack([])})
                roi["head"] = [int(x) for x in pp.head]
                if o["headcap"]:
                    v["head_cap"] = "ok" if p_cap[0] >= .5 else "violation"; conf["head_cap"] = float(max(p_cap[0], 1-p_cap[0]))
                if o["mask"] == "head" and facing:  # the old shared head crop, packed trunk
                    t = .6
                    v["mask"] = "ok" if p_mask[0] >= t else "violation"; conf["mask"] = float(max(p_mask[0], 1-p_mask[0]))
                elif o["mask"] != "off" and not facing:
                    v["mask"] = "unknown"          # back of the head: the mouth cannot be seen
                elif o["mask"] == "face" and M["mask"] is not None and pp.face is not None:
                    fc = _crop(im, pp.face, GEOMETRY.face_crop_px)
                    if fc is not None:
                        pm = float(M["mask"].run(None, {"face_crops": _stack([fc])})[0][0]); t = .30
                        v["mask"] = "ok" if pm >= t else "violation"; conf["mask"] = max(pm, 1-pm)
                        roi["face"] = [int(x) for x in pp.face]
        if o["gloves"] == "detector":
            st, c, bb = detector_gloves(M, im, pp.person_box, o)
            v["gloves"] = st
            if c: conf["gloves"] = c
            if bb: roi["gloves"] = bb
        elif o["gloves"] == "trunk":
            hands, hb = [], []
            for h in pp.hands[:2]:
                if o["gate"] and not roi_usable(h, im.shape, part="hand", others=pboxes, self_box=pp.person_box):
                    continue
                c = _crop(im, h, GEOMETRY.hand_crop_px)
                if c is not None: hands.append(c); hb.append(h)
            if len(hands) >= 2:
                _, _, g = M["packed"].run(None, {"head_crops": _stack([]), "hand_crops": _stack(hands[:2])})
                pr = float(min(g))
                v["gloves"] = "ok" if pr >= .5 else "violation"; conf["gloves"] = max(pr, 1-pr)
                roi["gloves"] = [int(min(b[0] for b in hb)), int(min(b[1] for b in hb)),
                                 int(max(b[2] for b in hb)), int(max(b[3] for b in hb))]
            else:
                v["gloves"] = "unknown"
        out.append((pp.person_box, v, conf, roi))
    return out


def detector_gloves(M, im, pb, o):
    H, W = im.shape[:2]; x1, y1, x2, y2 = pb
    pw, ph = x2-x1, y2-y1
    cx1, cy1 = int(max(0, x1-.18*pw)), int(max(0, y1-.18*ph))
    cx2, cy2 = int(min(W, x2+.18*pw)), int(min(H, y2+.18*ph))
    crop = im[cy1:cy2, cx1:cx2]
    if crop.size == 0 or min(crop.shape[:2]) < 16: return "unknown", None, None
    x, r, dx, dy = _letterbox(crop, 416)
    out = M["gloves"].run(None, {M["gloves"].get_inputs()[0].name: x})[0][0]
    best = {}
    for row in out:
        sc, cl = float(row[4]), int(row[5])
        if sc < o["det_conf"]: continue
        nm = DET_NAMES[cl] if 0 <= cl < len(DET_NAMES) else None
        if nm in ("gloves", "no_gloves") and sc > best.get(nm, (0, None))[0]:
            b = [(row[0]-dx)/r+cx1, (row[1]-dy)/r+cy1, (row[2]-dx)/r+cx1, (row[3]-dy)/r+cy1]
            best[nm] = (sc, [int(v) for v in b])
    pos, neg = best.get("gloves"), best.get("no_gloves")
    if pos and neg: return ("ok", pos[0], pos[1]) if pos[0] >= neg[0] else ("violation", neg[0], neg[1])
    if pos: return "ok", pos[0], pos[1]
    if neg: return "violation", neg[0], neg[1]
    return "unknown", None, None


# ------------------------------------------------------------------ drawing
def label(im, txt, x, y, col):
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)
    y = max(th + 2, y)
    cv2.rectangle(im, (x, y-th-3), (x+tw+4, y+2), (20, 20, 20), -1)
    cv2.putText(im, txt, (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, .45, col, 1, cv2.LINE_AA)


def draw_prod(im, dets, o):
    for box, nm, sc in dets:
        if nm == "person" and not o["prod_person"]: continue
        if nm != "person" and not o["prod_ppe"]: continue
        c = WHITE if nm == "person" else (RED if nm.startswith("no_") else GREEN)
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 1 if nm == "person" else 2)
        label(im, f"{nm} {sc:.2f}", x1, y1-3, c)
    return im


def draw_ours(im, people, o):
    for pb, v, conf, roi in people:
        x1, y1, x2, y2 = [int(t) for t in pb]
        if o["ours_person"]:
            cv2.rectangle(im, (x1, y1), (x2, y2), WHITE, 1)
        if o["show_roi"]:
            for k, b in roi.items():
                cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (200, 170, 60), 1)
        yy = y1 + 2
        for g in ("head_cap", "mask", "gloves"):
            if g not in v: continue
            c = COL[v[g]]
            txt = f"{g}: {v[g]}" + (f" ({conf[g]:.2f})" if g in conf else "")
            label(im, txt, x1, yy + 12, c); yy += 15
    return im


def pane(im, title, ms, n, w):
    h = int(im.shape[0] * w / im.shape[1])
    im = cv2.resize(im, (w, h))
    bar = np.full((30, w, 3), 24, np.uint8)
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, WHITE, 1, cv2.LINE_AA)
    t = f"{ms:5.0f} ms   {n} person{'s' if n != 1 else ''}"
    cv2.putText(bar, t, (w-190, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (170, 220, 170), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


_VIDEO_CACHE = {}


def probe(path):
    """Frames, fps and size, cached by (path, mtime) so the listing stays quick.

    The edge recorder leaves one-frame stubs behind, and a copy in progress
    leaves hidden temp files. Neither should reach the dropdown as if it were a
    recording, so both are reported and the page greys them out.
    """
    key = (str(path), path.stat().st_mtime_ns)
    if key in _VIDEO_CACHE: return _VIDEO_CACHE[key]
    c = cv2.VideoCapture(str(path))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); fps = c.get(cv2.CAP_PROP_FPS) or 0
    w, h = int(c.get(3)), int(c.get(4)); ok, _ = c.read(); c.release()
    info = {"name": path.name, "frames": n, "fps": round(fps, 1),
            "size_mb": round(path.stat().st_size/1e6, 1), "res": f"{w}x{h}",
            "secs": round(n/fps, 1) if fps else 0, "playable": bool(ok and n > 10),
            "dir": str(path.parent.relative_to(VIDEO_DIR)) if path.parent != VIDEO_DIR else ""}
    _VIDEO_CACHE[key] = info
    return info


def find_videos():
    out = []
    for p in sorted(VIDEO_DIR.rglob("*")):
        if p.suffix.lower() not in (".mp4", ".avi", ".mkv"): continue
        if p.name.startswith("."): continue            # rsync temp file, still copying
        out.append(p)
    return out


@app.route("/api/videos")
def videos():
    return jsonify([probe(p) for p in find_videos()])


def opts_from_request():
    q = request.args
    b = lambda k, d="1": q.get(k, d) == "1"
    return {"pose_conf": float(q.get("pose_conf", .40)), "det_conf": float(q.get("det_conf", .25)),
            "prod_conf": float(q.get("prod_conf", .30)), "gate": b("gate"),
            "headcap": b("headcap"), "mask": q.get("mask", "face"), "facing_gate": b("facing_gate"), "gloves": q.get("gloves", "detector"),
            "ours_person": b("ours_person"), "prod_person": b("prod_person"), "prod_ppe": b("prod_ppe"),
            "show_roi": b("show_roi", "0"), "show_prod": b("show_prod"), "show_ours": b("show_ours"),
            "stride": max(1, int(q.get("stride", 3))), "width": int(q.get("width", 720))}


@app.route("/stream")
def stream():
    name = request.args.get("video", "")
    o = opts_from_request()
    path = (VIDEO_DIR / name).resolve()
    if VIDEO_DIR.resolve() not in path.parents: return Response("outside the video directory", status=400)
    if not path.exists():
        return Response("no such video", status=404)

    def gen():
        M = models()
        cap = cv2.VideoCapture(str(path))
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); i = 0; continue
                i += 1
                if i % o["stride"]: continue
                panes = []
                if o["show_prod"]:
                    t0 = time.perf_counter()
                    d = prod_detect(M["prod"], frame, o["prod_conf"])
                    ms = (time.perf_counter()-t0)*1000
                    npers = sum(1 for _, nm, _ in d if nm == "person")
                    panes.append(pane(draw_prod(frame.copy(), d, o), "PRODUCTION  apd_mbg", ms, npers, o["width"]))
                if o["show_ours"]:
                    t0 = time.perf_counter()
                    p = ours_detect(M, frame, o)
                    ms = (time.perf_counter()-t0)*1000
                    panes.append(pane(draw_ours(frame.copy(), p, o), "OURS  pose + head/face trunks + crop-detector", ms, len(p), o["width"]))
                if not panes: panes = [pane(frame, "nothing selected", 0, 0, o["width"])]
                out = np.hstack(panes) if len(panes) > 1 else panes[0]
                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok: continue
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        finally:
            cap.release()
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


if __name__ == "__main__":
    print(f"videos from {VIDEO_DIR}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
