"""Lay the device's own output out like the replay, so the two compare folder by folder.

    python scripts/107_device_to_replay_layout.py --device <prod_out folder>
        [--out artifacts/replay/prod_device] [--camera masak] [--offset 0]

The device writes apd_storage/apd_events.db and apd_storage/snapshots/<camera>/<event_id>.jpg,
with timestamps in UTC. Those timestamps are when the device PROCESSED the frame,
not when it was filmed: matching device snapshots to the recordings by picture
found the device 0-23 s behind (typically ~18 s), varying event to event --
stream buffering and processing backlog. A fixed offset cannot fix that, so each
event is located in the recordings by its picture: the person box from the
device record is compared (normalised correlation) against recording frames
from 45 s before to 5 s after the device timestamp, 5 frames per second. The replay writes the same files but groups by recording
clip. This assigns every device event to the clip that was being recorded at
that moment (clip name = start time in UTC, length from the file) and writes:

    <out>/apd_storage/snapshots/<clip>/<event_id>.jpg
    <out>/apd_storage/snapshots/_not_recorded/<event_id>.jpg   (no clip covers it)
    <out>/apd_storage/apd_events.db    same schema; camera_id = clip name
    <out>/progress.jsonl               per clip, like the replay's
    <out>/alignment.csv                device time, filmed time, lag, match score
    <out>/summary.md

An event is placed by its picture when the best match scores >= --min-score and
beats the best match elsewhere in the window by --margin; otherwise it falls
back to its timestamp and is marked "unaligned".

Nothing in the device folder is modified. --offset (seconds) corrects a clock
difference between the device and the recorder.
"""
import argparse, calendar, collections, json, re, shutil, sqlite3, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--out", default=str(ROOT/"artifacts/replay/prod_device"))
ap.add_argument("--camera", default=None)
ap.add_argument("--offset", type=float, default=0.0)
ap.add_argument("--no-align", action="store_true", help="place events by timestamp only")
ap.add_argument("--min-score", type=float, default=0.8)
ap.add_argument("--margin", type=float, default=0.05)
ap.add_argument("--before", type=float, default=45.0, help="seconds before the device time to search")
ap.add_argument("--after", type=float, default=5.0)
a = ap.parse_args()
dev, out = Path(a.device), Path(a.out)
if out.exists(): shutil.rmtree(out)
snap_root = out/"apd_storage/snapshots"; snap_root.mkdir(parents=True)

clips = []
for p in sorted((ROOT/"data/recording").rglob("*")):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", p.name)
    if p.suffix.lower() != ".mp4" or not m or p.name.startswith("."): continue
    c = cv2.VideoCapture(str(p)); n = c.get(cv2.CAP_PROP_FRAME_COUNT); fps = c.get(cv2.CAP_PROP_FPS) or 25; c.release()
    if n > 10:
        s0 = calendar.timegm(tuple(int(x) for x in m.groups()) + (0, 0, 0))
        clips.append((s0, s0 + n/fps, p.stem, str(p.relative_to(ROOT/"data/recording"))))
clips.sort()


import numpy as np


def _crop(im, b):
    x1, y1, x2, y2 = b; w, h = x2-x1, y2-y1
    x1, x2 = int(x1+.12*w), int(x2-.12*w); y1, y2 = int(y1+.12*h), int(y2-.12*h)   # inside the drawn box
    c = im[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    return None if c.size == 0 else cv2.cvtColor(cv2.resize(c, (48, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _ncc(a, b):
    a = a-a.mean(); b = b-b.mean(); return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-6))


def locate(img, box, t):
    """(filmed time, clip stem, clip rel, score, margin) of the recording frame that best
    matches the device snapshot's person box, or None."""
    cd = _crop(img, box)
    if cd is None: return None
    scores = []
    for s0, s1, stem, rel in clips:
        lo, hi = t - a.before, t + a.after
        if s1 < lo or s0 > hi: continue
        cap = cv2.VideoCapture(str(ROOT/"data/recording"/rel))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        start = max(0.0, lo - s0)
        cap.set(cv2.CAP_PROP_POS_MSEC, start*1000)
        i = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        step = max(1, int(round(fps/5)))
        while cap.grab():
            vt = s0 + i/fps
            if vt > hi: break
            if i % step == 0 and vt >= lo:
                ok, fr = cap.retrieve()
                cr = _crop(fr, box) if ok else None
                if cr is not None: scores.append((_ncc(cd, cr), vt, stem, rel))
            i += 1
        cap.release()
    if not scores: return None
    scores.sort(reverse=True)
    best = scores[0]
    other = next((x[0] for x in scores if abs(x[1]-best[1]) > 3), 0.0)
    return best[1], best[2], best[3], best[0], best[0]-other


def clip_at(t):
    for s0, s1, stem, rel in clips:
        if s0 <= t <= s1: return stem, rel
    return None, None


src_db = next(dev.rglob("apd_events.db"))
src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
images = {p.stem: p for p in dev.rglob("*.jpg")}
dst = sqlite3.connect(out/"apd_storage/apd_events.db")
for (sql,) in src.execute("select sql from sqlite_master where type='table' and name in ('events','apd_detail')"):
    dst.execute(sql)
ecols = [r[1] for r in src.execute("PRAGMA table_info(events)")]
dcols = [r[1] for r in src.execute("PRAGMA table_info(apd_detail)")]

per_clip = collections.Counter(); cnt = collections.Counter(); align_rows = []
import csv
for row in src.execute(f"select {','.join(ecols)} from events order by timestamp"):
    e = dict(zip(ecols, row))
    if a.camera and e["camera_id"] != a.camera: cnt["other camera"] += 1; continue
    t = calendar.timegm(time.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S")) + a.offset
    img = images.get(e["event_id"])
    loc = None
    if not a.no_align and img is not None:
        im = cv2.imread(str(img))
        if im is not None:
            loc = locate(im, json.loads(e["person_bbox"] or "[0,0,0,0]"), t)
    if loc and loc[3] >= a.min_score and loc[4] >= a.margin:
        vt, stem, rel, score, margin = loc
        how = "by picture"
        align_rows.append([e["event_id"], e["timestamp"], time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(vt)) + f".{int((vt%1)*10)}",
                           stem, round(vt - [c[0] for c in clips if c[2] == stem][0], 1), round(t - vt, 1), round(score, 3), how])
    else:
        stem, rel = clip_at(t)
        how = "unaligned"
        align_rows.append([e["event_id"], e["timestamp"], "", stem or "", "", "",
                           round(loc[3], 3) if loc else "", how])
    folder = stem or "_not_recorded"
    if how == "by picture":            # same clock as the replay: when the frame was filmed
        e["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(vt))
    cnt[("in a recording" if stem else "not recorded") + f", {how}"] += 1
    if stem: per_clip[rel] += 1
    (snap_root/folder).mkdir(exist_ok=True)
    if img is not None:
        shutil.copy2(img, snap_root/folder/img.name)
    else:
        cnt["snapshot missing"] += 1
    e["device_camera_id"] = e["camera_id"]
    e["camera_id"] = folder
    e["image_path"] = f"apd_storage/snapshots/{folder}/{e['event_id']}.jpg"
    dst.execute(f"insert into events ({','.join(ecols)}) values ({','.join('?'*len(ecols))})", [e[k] for k in ecols])
    for d in src.execute(f"select {','.join(dcols)} from apd_detail where event_id=?", (e["event_id"],)):
        dst.execute(f"insert into apd_detail ({','.join(dcols)}) values ({','.join('?'*len(dcols))})", d)
dst.commit()

with (out/"alignment.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["event_id", "device_time_utc", "filmed_time_utc", "clip", "clip_offset_s", "lag_s", "match_score", "placed"])
    w.writerows(align_rows)
lags = [r[5] for r in align_rows if r[7] == "by picture"]

with (out/"progress.jsonl").open("w") as f:
    for s0, s1, stem, rel in clips:
        f.write(json.dumps({"clip": rel, "snapshots": per_clip.get(rel, 0), "ok": True, "source": "device"}) + "\n")

first = time.strftime("%Y-%m-%d %H:%M", time.gmtime(clips[0][0])) if clips else "-"
last = time.strftime("%Y-%m-%d %H:%M", time.gmtime(clips[-1][1])) if clips else "-"
L = ["# Device output in replay layout", "", f"source: `{src_db}`" + (f", camera `{a.camera}`" if a.camera else ""),
     f"recordings cover {first} to {last} UTC ({len(clips)} clips); clock offset {a.offset}s", "",
     "| | events |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cnt.items()] + \
    ["", f"{sum(1 for v in per_clip.values() if v)} clips have at least one device event."] + \
    ([f"Device lag behind the filmed frame, events placed by picture: median {np.median(lags):.1f} s, "
      f"range {min(lags):.1f} to {max(lags):.1f} s (n={len(lags)})."] if lags else [])
(out/"summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L)); print(f"\nwrote {out}")
