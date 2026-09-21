"""Replay every playable recording in data/recording through both monitors.

    python scripts/105_replay_recordings.py            # both arms, in parallel
    python scripts/105_replay_recordings.py ours       # one arm

Output per arm, laid out like a device's storage:
    artifacts/replay/<arm>/apd_storage/snapshots/<clip>/<event>.jpg
    artifacts/replay/<arm>/apd_storage/apd_events.db   (camera_id = clip name)
    artifacts/replay/<arm>/logs/                       (monitor logs)
    artifacts/replay/<arm>/progress.jsonl              (one line per clip; used to resume)

Event timestamps are processing time, as production writes them; the clip is in
camera_id. Resumes: clips already in progress.jsonl are skipped.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/(os.environ.get("APD_REPLAY_OUT") or "artifacts/replay")
SRC = ROOT/(os.environ.get("APD_RECORDINGS") or "data/recording")     # folder of clips to replay
ONLY = None                                                           # APD_CLIPS=file.txt -> only these stems
if os.environ.get("APD_CLIPS"):
    ONLY = {l.strip() for l in open(os.environ["APD_CLIPS"]) if l.strip()}
PY = sys.executable
arms = sys.argv[1:] or ["ours", "prod"]


def clips():
    out = []
    for p in sorted(SRC.rglob("*")):
        if p.suffix.lower() not in (".mp4", ".avi", ".mkv") or p.name.startswith("."): continue
        if ONLY is not None and p.stem not in ONLY: continue
        c = cv2.VideoCapture(str(p)); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); ok, _ = c.read(); c.release()
        if ok and n > 10: out.append(p)
    return out


def run_arm(arm, todo):
    d = OUT/arm; d.mkdir(parents=True, exist_ok=True)
    prog = d/"progress.jsonl"
    done = {json.loads(l)["clip"] for l in prog.read_text().splitlines()} if prog.exists() else set()
    for i, clip in enumerate(todo, 1):
        rel = str(clip.relative_to(SRC))
        if rel in done: continue
        t0 = time.time()
        snaps = d/"apd_storage/snapshots"/clip.stem
        before = len(list(snaps.glob("*.jpg"))) if snaps.exists() else 0
        r = subprocess.run([PY, str(ROOT/"scripts/105_replay_one.py"), arm, str(clip), str(d)],
                           cwd=d, capture_output=True, text=True, timeout=4*3600)
        after = len(list(snaps.glob("*.jpg"))) if snaps.exists() else 0
        # production's shutdown calls cv2.destroyAllWindows(), which fails headless
        # after all work is done; any other traceback is a real failure
        tb = [l for l in r.stderr.splitlines() if "Error" in l and "cvDestroyAllWindows" not in l
              and "window.cpp" not in l and "Warning" not in l]
        rec = {"clip": rel, "snapshots": after-before, "seconds": round(time.time()-t0, 1),
               "ok": not tb, "error": tb[-1][:300] if tb else None}
        with prog.open("a") as f: f.write(json.dumps(rec) + "\n")
        print(f"[{arm}] {i}/{len(todo)} {rel}: {rec['snapshots']} snapshots in {rec['seconds']}s"
              + ("" if rec["ok"] else f"  ERROR {rec['error']}"), flush=True)


if __name__ == "__main__":
    todo = clips()
    print(f"{len(todo)} playable clips; arms {arms}", flush=True)
    with ThreadPoolExecutor(len(arms)) as ex:
        list(ex.map(lambda a: run_arm(a, todo), arms))
    print("ALL DONE", flush=True)
