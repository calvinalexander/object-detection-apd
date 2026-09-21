"""Run the deployed APD detector over a video, off-device.

Pre- and post-processing are copied from apd_v3.axnet so results are comparable
with what the edge device produces:

  preprocess   letterbox to 640x640, pad 114, scale_up, RGB, NCHW, /255
  postprocess  YOLOv8 decode, 7 classes, then class-aware NMS at 0.7, max 300

The AIPU build cannot run here -- pool_ddr_const.bin is a Metis weight pool -- so
this takes the source ONNX. That difference is itself useful: this is the FP32
reference the INT8 device build should be compared against.
"""
import argparse, collections, json, sys, time
from pathlib import Path
import cv2, numpy as np

LABELS = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
COLOURS = {"person": (240, 190, 60), "gloves": (90, 220, 120), "head_cap": (120, 200, 255),
           "mask": (200, 160, 255), "no_gloves": (60, 60, 240), "no_head_cap": (60, 120, 240),
           "no_mask": (80, 60, 220)}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True,
                help="a .onnx file, or an OpenVINO .xml (Ultralytics export)")
ap.add_argument("--video", required=True)
ap.add_argument("--out", default="artifacts/apd_demo.mp4")
ap.add_argument("--conf", type=float, default=0.30, help="matches APD_GROUP_THRESH in config_apd.py")
ap.add_argument("--nms", type=float, default=0.7, help="from apd_v3.axnet postprocess1")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--tag", default="apd", help="label burned into the output video")
args = ap.parse_args()

# Two backends, because the deployed models arrive in two portable formats:
# an ONNX export, or an Ultralytics OpenVINO directory. The AIPU .axnet builds
# are not portable and cannot be served here at all.
if args.model.endswith(".xml"):
    import openvino as ov
    core = ov.Core()
    dev = "GPU" if "GPU" in core.available_devices else "CPU"
    compiled = core.compile_model(core.read_model(args.model), dev)
    out_port = compiled.outputs[0]
    def infer(x): return compiled([x])[out_port]
    runtime = f"openvino/{dev}"
    in_name = "x"
else:
    import onnxruntime as ort
    sess = ort.InferenceSession(args.model,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    def infer(x): return sess.run(None, {in_name: x})[0]
    runtime = sess.get_providers()[0]
print(f"model  : {Path(args.model).name}")
print(f"runtime: {runtime}")

def letterbox(img, size):
    """Matches libtransform_resize with padding:114 letterbox:1 scale_up:1."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top

def decode(raw, r, dx, dy, conf_thr):
    """YOLOv8 head: (1, 4+nc, anchors) -> xyxy in original-image coordinates."""
    p = raw[0]
    if p.shape[0] < p.shape[1]:
        p = p.T                                  # -> (anchors, 4+nc)
    boxes_cxcywh, scores = p[:, :4], p[:, 4:4 + len(LABELS)]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(scores)), cls]
    keep = conf >= conf_thr
    if not keep.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    b, conf, cls = boxes_cxcywh[keep], conf[keep], cls[keep]
    xy = np.empty_like(b)
    xy[:, 0] = b[:, 0] - b[:, 2] / 2; xy[:, 1] = b[:, 1] - b[:, 3] / 2
    xy[:, 2] = b[:, 0] + b[:, 2] / 2; xy[:, 3] = b[:, 1] + b[:, 3] / 2
    xy[:, [0, 2]] -= dx; xy[:, [1, 3]] -= dy
    xy /= r
    return xy, conf, cls

cap = cv2.VideoCapture(args.video)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS / args.stride, (W, H))
print(f"video  : {W}x{H} @{FPS:.0f}fps, {total} frames -> {args.out}\n")

counts, per_frame, latency, n = collections.Counter(), [], [], 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if n % args.stride:
        n += 1; continue
    img, r, dx, dy = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), args.imgsz)
    x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    t0 = time.time()
    raw = infer(x)
    latency.append((time.time() - t0) * 1000)
    boxes, conf, cls = decode(raw, r, dx, dy, args.conf)
    if len(boxes):
        idx = cv2.dnn.NMSBoxes(
            [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
            conf.tolist(), args.conf, args.nms)
        idx = np.array(idx).flatten() if len(idx) else np.zeros(0, int)
        boxes, conf, cls = boxes[idx], conf[idx], cls[idx]
    seen = collections.Counter()
    for b, c, k in zip(boxes, conf, cls):
        name = LABELS[k]; counts[name] += 1; seen[name] += 1
        col = COLOURS[name]
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        tag = f"{name} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
        cv2.putText(frame, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    per_frame.append(sum(seen.values()))
    cv2.rectangle(frame, (0, 0), (W, 26), (22, 28, 32), -1)
    cv2.putText(frame, f"{args.tag}  frame {n}  " +
                "  ".join(f"{k}:{v}" for k, v in sorted(seen.items())),
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 240, 242), 1)
    writer.write(frame)
    n += 1
    if n % 200 == 0: print(f"  {n}/{total}", flush=True)
    if args.max_frames and n >= args.max_frames: break

cap.release(); writer.release()
proc = len(per_frame)
print(f"\nprocessed {proc} frames, {np.mean(latency):.1f} ms/frame "
      f"({1000/np.mean(latency):.1f} fps on {runtime})")
print(f"\n{'class':<14}{'detections':>11}{'per frame':>11}")
for k in LABELS:
    print(f"{k:<14}{counts[k]:>11}{counts[k]/max(proc,1):>11.2f}")
print(f"\nframes with no detection at all: {sum(1 for v in per_frame if v==0)}/{proc}")
print(f"wrote {args.out}")
