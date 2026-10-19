"""Render a run as an annotated MP4: the same overlay as the tuner (boxes coloured by fill, bold box when counted
OUT, VIOLATION tag, running counts, counting line), for one version or two side by side.

Usage: python render_run.py <source video> <job.json> <out.mp4>

job.json (written by server.mjs from what the browser counted with the current settings):
  {fps, line: [[x1, y1], [x2, y2]],
   panels: [{label: "EXISTING" | "PATCHED" | null,
             frames: [[[id, x1, y1, x2, y2, cls], ...] per frame],      # tracked boxes seen in that frame
             events: [{k, dir, viol, oid}, ...]}]}                         # counted crossings
The source is decoded with the same scale/pad/fps filter as process_run.py, so frame k matches detections frame k.
"""
import json, subprocess, sys
import cv2, numpy as np

W, H, FPS = 1280, 720, 15
SRC, JOB, OUT = sys.argv[1:4]
job = json.load(open(JOB))
fps = job.get("fps", FPS)

def bgr(hexcol):
    h = hexcol.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

# same colours as src/settings.ts: green (full) .. red (empty); a stack is green
COLOURS = {0: bgr("#ef3b2c"), 1: bgr("#1fbf4a"), 2: bgr("#f2c418"), 3: bgr("#f47c20"), 4: bgr("#8fd13a"), 5: bgr("#1fbf4a")}
LINE, IN_COL, DARK, VIOL = bgr("#ffb020"), bgr("#3fb8e0"), bgr("#1a1206"), bgr("#e0261b")
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text_box(img, text, org, scale, fg, bg, thick=1, pad=5):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - th - pad), (x + tw + 2 * pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, text, (x + pad, y), FONT, scale, fg, thick, cv2.LINE_AA)


def draw_line(img, line):
    (x1, y1), (x2, y2) = line
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.line(img, p1, p2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, p1, p2, LINE, 2, cv2.LINE_AA)
    L = max(np.hypot(x2 - x1, y2 - y1), 1)
    n = np.array([-(y2 - y1) / L, (x2 - x1) / L])               # points to the OUT side
    m = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    tip = m + n * 40
    cv2.arrowedLine(img, tuple(m.astype(int)), tuple(tip.astype(int)), LINE, 2, cv2.LINE_AA, tipLength=0.3)
    for label, pos, bg, fg in (("OUT", m + n * 64, LINE, DARK), ("IN", m - n * 28, IN_COL, (28, 22, 6))):
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        text_box(img, label, (int(pos[0] - tw / 2 - 4), int(pos[1] + th / 2)), 0.45, fg, bg, 1, 4)
    for i, p in enumerate((p1, p2)):
        cv2.circle(img, p, 8, LINE, -1, cv2.LINE_AA)
        cv2.circle(img, p, 8, DARK, 1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (p[0] - 4, p[1] + 4), FONT, 0.38, DARK, 1, cv2.LINE_AA)


def draw_panel(img, panel, k, line):
    draw_line(img, line)                                          # first, so boxes and VIOLATION tags sit on top
    flash = {e["oid"]: e for e in panel["events"] if e["k"] <= k < e["k"] + 22}
    snap = panel["frames"][k] if k < len(panel["frames"]) else []
    for oid, x1, y1, x2, y2, cls in snap:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        f = flash.get(oid)
        counted = f is not None                                   # bold: OUT in its fill colour, IN in blue
        col = IN_COL if counted and f["dir"] == "in" else COLOURS.get(cls, (200, 200, 200))
        if counted:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 6)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 4 if counted else 2)
        if f is not None and f["viol"]:
            text_box(img, "VIOLATION", (x1 - 2, y1 - 6), 0.5, (255, 255, 255), VIOL, 1, 4)
    o = i = v = 0
    for e in panel["events"]:
        if e["k"] > k:
            break
        o += e["dir"] == "out"; i += e["dir"] == "in"; v += bool(e["viol"])
    text_box(img, f"OUT {o}   IN {i}   VIOLATIONS {v}", (12, 34), 0.62, (255, 255, 255), (20, 20, 20), 1, 7)
    if panel.get("label"):
        lab = panel["label"]
        (tw, _), _ = cv2.getTextSize(lab, FONT, 0.7, 2)
        text_box(img, lab, (W - tw - 30, 36), 0.7, (255, 255, 255), (131, 115, 10) if lab == "PATCHED" else (40, 40, 40), 2, 8)
    return img


def draw_expected(img, k, exp):
    """The expected running count at frame k: one line centred at the bottom, in the panels' own count style."""
    t = k / fps
    done = [c for c in exp["crossings"] if c["t"] <= t]
    o = sum(c["dir"] == "out" for c in done); i = len(done) - o; v = sum(bool(c["viol"]) for c in done)
    text = f"EXPECTED   OUT {o}   IN {i}   VIOLATIONS {v}"
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    text_box(img, text, (img.shape[1] // 2 - tw // 2 - 7, img.shape[0] - 22), 0.55, (255, 255, 255), (20, 20, 20), 1, 7)


def main():
    vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={fps}"
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", SRC, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    panels = job["panels"]
    two = len(panels) == 2
    exp = job.get("expected") if two else None
    ow, oh = (1920, 540) if two else (W, H)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
                            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "22", "-preset", "fast", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
    n = max(len(p["frames"]) for p in panels)
    k = 0
    while k < n:
        raw = dec.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:
            break
        frame = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
        if two:
            a = cv2.resize(draw_panel(frame.copy(), panels[0], k, job["line"]), (958, 540), interpolation=cv2.INTER_AREA)
            b = cv2.resize(draw_panel(frame.copy(), panels[1], k, job["line"]), (958, 540), interpolation=cv2.INTER_AREA)
            out = np.hstack([a, np.full((540, 4, 3), 255, np.uint8), b])
            if exp:
                draw_expected(out, k, exp)
        else:
            out = draw_panel(frame.copy(), panels[0], k, job["line"])
        enc.stdin.write(np.ascontiguousarray(out).tobytes())
        k += 1
    dec.stdout.close(); dec.wait()
    enc.stdin.close()
    if enc.wait() != 0 or k == 0:
        sys.exit("Rendering failed: ffmpeg could not encode the video.")
    print(f"rendered {k} frames", flush=True)


if __name__ == "__main__":
    main()
