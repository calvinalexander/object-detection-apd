"""Side-by-side render: production (apd_mbg stand-in) left, ours right.

  APD_BUNDLE=deploy/apd_person renders the one-network staging build instead of the
  deployed bundle; set OUT_SUFFIX to keep both (e.g. OUT_SUFFIX=_person).

  python scripts/109_render_side_by_side.py 2026-09-16_00-07-46 [more clips]
  python scripts/109_render_side_by_side.py <clip> --stitch-only   # reuse rendered arms

Each arm runs through scripts/105_replay_one.py with REPLAY_VIDEO set, so the
frames carry the same draw_table annotation as the snapshots. Output:
artifacts/replay/renders/<clip>/{prod,ours}_<clip>.mp4 and side_by_side.mp4.
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SUF = os.environ.get("OUT_SUFFIX", "")
SRC = os.environ.get("APD_RECORDINGS", "data/recording")      # folder holding the clips
LABEL = {"prod": "PRODUCTION (apd_mbg)",
         "ours": "OURS (one-network person model)" if os.environ.get("APD_BUNDLE", "").endswith("apd_person") else "OURS"}


def run_arm(clip, arm):
    out = ROOT/f"artifacts/replay/renders/{clip}{SUF}"; (out/f"_run_{arm}").mkdir(parents=True, exist_ok=True)
    vid = out/f"{arm}_{clip}.mp4"
    with open(out/f"_run_{arm}.log", "w") as log:
        subprocess.run([PY, str(ROOT/"scripts/105_replay_one.py"), arm, str(ROOT/SRC/f"{clip}.mp4"), str(out/f"_run_{arm}")],
                       env={**__import__("os").environ, "REPLAY_VIDEO": str(vid)}, stdout=log, stderr=subprocess.STDOUT)
    return found(clip, arm)   # exit code ignored: production's destroyAllWindows fails headless at the very end


def found(clip, arm):
    # the monitor appends the camera id to the output name: <arm>_<clip>_<clip>.mp4
    out = ROOT/f"artifacts/replay/renders/{clip}{SUF}"
    return max(out.glob(f"{arm}_{clip}*.mp4"), key=lambda q: len(q.name))


def tag(f, text):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(f, (0, f.shape[0]-h-14), (w+12, f.shape[0]), (0, 0, 0), -1)
    cv2.putText(f, text, (6, f.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def stitch(clip, vids):
    out = ROOT/f"artifacts/replay/renders/{clip}{SUF}/side_by_side.mp4"
    caps = {a: cv2.VideoCapture(str(v)) for a, v in vids.items()}
    fps = caps["prod"].get(cv2.CAP_PROP_FPS) or 10
    enc = None; n = 0
    while True:
        fr = {}
        for a, c in caps.items():
            ok, f = c.read()
            if not ok: break
            tag(f, LABEL[a]); fr[a] = f
        if len(fr) < 2: break
        img = np.hstack([fr["prod"], fr["ours"]])
        if enc is None:
            enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                    "-s", f"{img.shape[1]}x{img.shape[0]}", "-r", str(fps), "-i", "-",
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], stdin=subprocess.PIPE)
        enc.stdin.write(img.tobytes()); n += 1
    if enc is None:
        raise SystemExit(f"{clip}: no frames read from {vids}")
    enc.stdin.close(); enc.wait()
    print(f"{clip}: {n} frames -> {out}")


if __name__ == "__main__":
    clips = [c for c in sys.argv[1:] if not c.startswith("--")]
    if "--stitch-only" in sys.argv:
        for c in clips: stitch(c, {a: found(c, a) for a in ("prod", "ours")})
        raise SystemExit
    with ThreadPoolExecutor(4) as ex:
        jobs = {(c, a): ex.submit(run_arm, c, a) for c in clips for a in ("prod", "ours")}
        vids = {k: j.result() for k, j in jobs.items()}
    for c in clips:
        stitch(c, {a: vids[(c, a)] for a in ("prod", "ours")})
