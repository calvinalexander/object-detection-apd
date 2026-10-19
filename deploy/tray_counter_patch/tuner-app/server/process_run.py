"""Turn an uploaded video into a tuner run: recorder-format video, web video, poster and per-frame detections.

Usage: python process_run.py <run dir>      (the dir holds upload.<ext>; written by server.mjs)

Writes into the run dir:
  work.mp4      1280x720 at 15 fps, what recorder.py records and what counter.py reads
  video.mp4     640x360 at 15 fps for the browser (frame k of video.mp4 = frame k of work.mp4)
  poster.jpg    a frame for the player
  bg.png        median background (the stand-in detector needs a fixed camera)
  data.json     {fps, w, h, dets: [[x1, y1, x2, y2, cls], ...] per frame, ref: []}
  status.json   {stage, done, total, error?}, rewritten as it goes; the server reads it for progress

Detector: the background-subtraction stand-in used for the s14 test (s14/det.py, s14/fill.py), with the one
s14-specific number (how big one tray looks at each image height) estimated from this video instead.
It only works for a fixed camera over a plain belt or counter, and it does not recognise stacks.
"""
import json, os, subprocess, sys, time, traceback
import cv2, numpy as np

FPS, W, H = 15, 1280, 720
CUTS = [(0.04, 0), (0.17, 3), (0.27, 2), (0.32, 4)]   # food share -> class (s14/detect_all.py); else 1 = full
RUN = sys.argv[1]
P = lambda name: os.path.join(RUN, name)


def status(stage, done=0, total=0, **extra):
    tmp = P("status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(dict(stage=stage, done=done, total=total, updated=time.time(), **extra), f)
    os.replace(tmp, P("status.json"))


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True, stderr=subprocess.PIPE)


def fill_class(fo):
    for c, k in CUTS:
        if fo < c:
            return k
    return 1


def fill_holes(m):
    """Fill everything enclosed by a blob (food on a tray leaves holes in the foreground)."""
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cs, -1, 1, cv2.FILLED)
    return out


def odd(v, lo=3):
    v = max(lo, int(round(v)))
    return v if v % 2 else v + 1


class Scale:
    """Kernel sizes and the smallest blob, all relative to one tray's size in pixels (sqrt of its area).
    At s14's scale (one tray about 30,000 px2) these come out as the sizes the s14 stand-in was tuned with."""
    def __init__(self, tray_area):
        k = float(np.sqrt(tray_area))
        self.tray_area = float(tray_area)
        self.close, self.open, self.erode = odd(0.05 * k), odd(0.04 * k), odd(0.1 * k)
        self.min_area = 0.25 * tray_area                                  # keeps trays half into the frame

    def __repr__(self):
        return f"tray {self.tray_area:.0f} px2: close {self.close}, open {self.open}, erode {self.erode}, min area {self.min_area:.0f}"


def foreground(img, bg_blur, close=5, open_=3, thr=40):
    d = cv2.absdiff(cv2.GaussianBlur(img, (5, 5), 0), bg_blur).max(axis=2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_, open_), np.uint8))
    return fill_holes(m)


def blobs(img, m, erode, min_area):
    """Seeds from an eroded mask (touching trays come apart), then watershed back to the full mask."""
    core = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))
    k, seeds = cv2.connectedComponents(core)
    markers = seeds.astype(np.int32) + 1
    markers[(m == 1) & (core == 0)] = 0
    markers[m == 0] = 1
    cv2.watershed(img, markers)
    out = []
    for lab in range(2, k + 1):
        mm = (markers == lab).astype(np.uint8)
        if mm.sum() >= min_area:
            out.append(mm)
    return out


def tray_size(frames, bg_blur):
    """Most common blob size with small fixed kernels: on a belt most blobs are single trays."""
    areas = []
    for f in frames:
        m = foreground(f, bg_blur)
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        for x, y, w, h, a in st[1:]:
            if a > 300 and x > 2 and y > 2 and x + w < W - 2 and y + h < H - 2:
                areas.append(a)
    if len(areas) < 10:
        raise RuntimeError("Too few moving objects were found to measure the tray size. Is the camera fixed?")
    la = np.log(areas)
    hist, edges = np.histogram(la, bins=np.arange(la.min(), la.max() + 0.1, 0.1))
    hist = np.convolve(hist, [1, 2, 3, 2, 1], "same")                       # smooth, then take the peak
    return float(np.exp(edges[np.argmax(hist)] + 0.05))


def split(mm, n):
    """Split a blob of n touching trays into n parts along its long axis (equal pixel counts)."""
    ys, xs = np.nonzero(mm)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mu = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    proj = (pts - mu) @ vt[0]
    lab = np.searchsorted(np.quantile(proj, np.linspace(0, 1, n + 1)[1:-1]), proj)
    parts = []
    for k in range(n):
        p = np.zeros_like(mm)
        p[ys[lab == k], xs[lab == k]] = 1
        parts.append(p)
    return parts


def fit_tray_area(samples):
    """One tray's area as a straight line in its centre height, fitted to the smaller blobs (merged ones fall out)."""
    a = np.array(samples, float)
    if len(a) < 8:
        return None
    keep = np.ones(len(a), bool)
    for _ in range(5):
        slope, icpt = np.polyfit(a[keep, 0], a[keep, 1], 1)
        pred = np.maximum(slope * a[:, 0] + icpt, 1)
        keep = a[:, 1] < 1.35 * pred
        if keep.sum() < 8:
            break
    return float(slope), float(icpt)


def main():
    src = next(f for f in os.listdir(RUN) if f.startswith("upload."))
    status("converting")
    # recorder format: 1280x720 at 15 fps, letterboxed if the source has another aspect ratio
    vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS}"
    ffmpeg("-i", P(src), "-vf", vf, "-an", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", P("work.mp4"))
    ffmpeg("-i", P("work.mp4"), "-vf", "scale=640:360", "-an", "-c:v", "libx264", "-crf", "28", "-preset", "fast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", P("video.mp4"))

    cap = cv2.VideoCapture(P("work.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 2:
        raise RuntimeError("The video has no readable frames.")
    ffmpeg("-ss", str(min(2.0, n / FPS / 2)), "-i", P("work.mp4"), "-frames:v", "1", "-vf", "scale=640:360", "-q:v", "4", P("poster.jpg"))

    status("background", 0, n)
    frames = []
    for i in np.linspace(0, n - 1, min(120, n)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    # a dark percentile, not the median: trays are brighter than the belt, and on a busy belt they cover each
    # spot more than half the time, so the median would bake trays into the background
    bg = np.percentile(np.stack(frames), 20, axis=0).astype(np.uint8)
    cv2.imwrite(P("bg.png"), bg)
    bg_blur = cv2.GaussianBlur(bg, (5, 5), 0)

    status("calibrating", 0, n)
    sample = frames[:: max(1, len(frames) // 40)]
    sc = Scale(tray_size(sample, bg_blur))
    print(sc, flush=True)
    samples = []
    for f in sample:
        for mm in blobs(f, foreground(f, bg_blur, sc.close, sc.open), sc.erode, sc.min_area):
            x, y, w, h = cv2.boundingRect(mm)
            if x > 2 and y > 2 and x + w < W - 2 and y + h < H - 2:          # skip trays cut by the frame edge
                samples.append((y + h / 2, int(mm.sum())))
    fit = fit_tray_area(samples)

    status("detecting", 0, n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    dets, k, t0 = [], 0, time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        fr = []
        for mm in blobs(img, foreground(img, bg_blur, sc.close, sc.open), sc.erode, sc.min_area):
            x, y, w, h = cv2.boundingRect(mm)
            parts = [mm]
            one = sc.tray_area                                          # one tray's area at this height
            if fit:
                f_one = fit[0] * (y + h / 2) + fit[1]
                if 0.5 * sc.tray_area < f_one < 2 * sc.tray_area:
                    one = f_one
            if True:
                parts_n = int(round(mm.sum() / one))
                if parts_n >= 2:
                    parts = split(mm, min(parts_n, 6))
            for p in parts:
                x, y, w, h = cv2.boundingRect(p)
                inner = cv2.erode(p, np.ones((9, 9), np.uint8)) > 0
                s, v = hsv[..., 1][inner], hsv[..., 2][inner]
                food = float(((s > 45) | ((v > 200) & (s <= 45))).mean()) if inner.any() else 0.0
                # extra values after the class (ignored by the counter): food share, area relative to one tray
                fr.append([x, y, x + w, y + h, fill_class(food), round(food, 3), round(float(p.sum()) / one, 2)])
        dets.append(fr)
        k += 1
        if k % 15 == 0:
            status("detecting", k, n, fps=round(k / max(time.time() - t0, 1e-6), 1))
    with open(P("data.json"), "w") as f:
        json.dump(dict(fps=FPS, w=W, h=H, dets=dets, ref=[], detector="stand-in (background subtraction)",
                       tray_area_fit=fit, scale=repr(sc)), f, separators=(",", ":"))
    if os.environ.get("TUNER_KEEP_WORK") != "1":                            # video.mp4 is enough for the browser
        os.remove(P("work.mp4"))
    status("ready", k, k)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        msg = e.stderr.decode(errors="replace")[-400:] if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
        status("error", error=msg or type(e).__name__)
        sys.exit(1)
