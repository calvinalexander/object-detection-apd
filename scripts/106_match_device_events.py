"""Match the device's own snapshots to replay events, event by event.

    python scripts/106_match_device_events.py --device <prod_out folder>
        [--replay artifacts/replay/prod] [--camera masak] [--tol 5] [--iou 0.3]
        [--offset 0] [--out artifacts/replay/match_prod]

The device writes apd_events.db (timestamp in UTC, camera, track, person box,
violations) and one snapshot per event named <event_id>.jpg. The replay writes
the same, stamped with the time the frame was filmed. So an event can be paired
with a replay event when:

  1. its timestamp falls inside a recording in data/recording
     (clip name = start time in UTC, length from the file) -- otherwise the
     replay never saw that moment and the event is set aside as "not recorded"
  2. a replay event lies within --tol seconds (--offset corrects a clock skew)
  3. the two person boxes overlap by at least --iou

Output in --out:
  matches.csv       one row per device event and per unmatched replay event
  summary.md        counts and violation agreement
  pairs/            device snapshot | replay snapshot, side by side
  device_only/      device events in recorded time the replay did not raise
  replay_only/      replay events with no device event
"""
import argparse, calendar, csv, json, re, shutil, sqlite3, time
from collections import Counter
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True, help="folder copied from the device (contains apd_events.db and snapshots)")
ap.add_argument("--replay", default=str(ROOT/"artifacts/replay/prod"))
ap.add_argument("--camera", default=None, help="device camera_id to keep, e.g. masak")
ap.add_argument("--tol", type=float, default=5.0)
ap.add_argument("--iou", type=float, default=0.3)
ap.add_argument("--offset", type=float, default=0.0, help="seconds added to device timestamps")
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev, rep = Path(a.device), Path(a.replay)
out = Path(a.out or ROOT/f"artifacts/replay/match_{rep.name}")
if out.exists(): shutil.rmtree(out)
for d in ("pairs", "device_only", "replay_only"): (out/d).mkdir(parents=True)


def ts(s): return calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


def iou(x, y):
    ix = max(0, min(x[2], y[2]) - max(x[0], y[0])); iy = max(0, min(x[3], y[3]) - max(x[1], y[1])); i = ix*iy
    return i / max(1e-6, (x[2]-x[0])*(x[3]-x[1]) + (y[2]-y[0])*(y[3]-y[1]) - i)


def events(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    return [dict(zip(("event_id", "t", "camera", "track", "box", "viol", "img"),
                     (r[0], ts(r[1]), r[2], r[3], json.loads(r[4] or "[0,0,0,0]"), sorted(json.loads(r[5] or "[]")), r[6])))
            for r in c.execute("select event_id,timestamp,camera_id,track_id,person_bbox,violation,image_path from events")]


# recorded time
clips = []
for p in sorted((ROOT/"data/recording").rglob("*.mp4")):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", p.name)
    if not m or p.name.startswith("."): continue
    cap = cv2.VideoCapture(str(p)); n = cap.get(cv2.CAP_PROP_FRAME_COUNT); fps = cap.get(cv2.CAP_PROP_FPS) or 25; cap.release()
    if n > 10:
        s0 = calendar.timegm(tuple(int(x) for x in m.groups()) + (0, 0, 0)); clips.append((s0, s0 + n/fps, p.stem))


def clip_at(t):
    for s0, s1, name in clips:
        if s0 <= t <= s1: return name, t - s0
    return None, None


dev_db = next(dev.rglob("apd_events.db"))
D = [e for e in events(dev_db) if a.camera is None or e["camera"] == a.camera]
R = events(rep/"apd_storage/apd_events.db")
done_clips = {json.loads(l)["clip"].rsplit(".", 1)[0].split("/")[-1] for l in (rep/"progress.jsonl").read_text().splitlines()} \
    if (rep/"progress.jsonl").exists() else {c[2] for c in clips}
snaps_dev = {p.stem: p for p in dev.rglob("*.jpg")}

rows, used = [], set()
cnt = Counter()
for e in sorted(D, key=lambda e: e["t"]):
    t = e["t"] + a.offset
    clip, off = clip_at(t)
    if clip is None: cnt["device, not recorded"] += 1; continue
    if clip not in done_clips: cnt["device, clip not replayed yet"] += 1; continue
    best = None
    for j, r in enumerate(R):
        if j in used or r["camera"] != clip or abs(r["t"] - t) > a.tol: continue
        o = iou(e["box"], r["box"])
        if o >= a.iou and (best is None or (o, -abs(r["t"]-t)) > (best[1], -best[2])): best = (j, o, abs(r["t"]-t))
    row = {"device_event": e["event_id"], "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)),
           "clip": clip, "clip_offset_s": round(off, 1), "device_violation": "+".join(e["viol"])}
    if best:
        j, o, dt = best; used.add(j); r = R[j]
        row.update(replay_event=r["event_id"], replay_violation="+".join(r["viol"]), dt_s=dt, iou=round(o, 2),
                   same_violation=e["viol"] == r["viol"])
        cnt["matched"] += 1; cnt["matched, same violation"] += e["viol"] == r["viol"]
        im1 = cv2.imread(str(snaps_dev.get(e["event_id"], ""))); im2 = cv2.imread(str(rep/r["img"]))
        if im1 is not None and im2 is not None:
            h = 480; pair = np.hstack([cv2.resize(x, (int(x.shape[1]*h/x.shape[0]), h)) for x in (im1, im2)])
            cv2.imwrite(str(out/"pairs"/f"{row['time_utc'].replace(' ', '_').replace(':', '-')}_{e['event_id'][:8]}.jpg"), pair)
    else:
        cnt["device only"] += 1
        if e["event_id"] in snaps_dev:
            shutil.copy(snaps_dev[e["event_id"]], out/"device_only"/f"{row['time_utc'].replace(' ', '_').replace(':', '-')}_{e['event_id'][:8]}.jpg")
    rows.append(row)
for j, r in enumerate(R):
    if j in used or r["camera"] not in done_clips: continue
    cnt["replay only"] += 1
    tu = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r["t"]))
    rows.append({"replay_event": r["event_id"], "time_utc": tu, "clip": r["camera"], "replay_violation": "+".join(r["viol"])})
    if (rep/r["img"]).exists():
        shutil.copy(rep/r["img"], out/"replay_only"/f"{tu.replace(' ', '_').replace(':', '-')}_{r['event_id'][:8]}.jpg")

keys = ["time_utc", "clip", "clip_offset_s", "device_event", "device_violation", "replay_event",
        "replay_violation", "dt_s", "iou", "same_violation"]
with (out/"matches.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in sorted(rows, key=lambda r: r["time_utc"])]
L = [f"# Device events vs replay ({rep.name})", "",
     f"device db: `{dev_db}`" + (f", camera `{a.camera}`" if a.camera else ""),
     f"tolerance {a.tol}s, person-box overlap >= {a.iou}, clock offset {a.offset}s", "",
     "| | count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cnt.items()]
(out/"summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L)); print(f"\nwrote {out}")
