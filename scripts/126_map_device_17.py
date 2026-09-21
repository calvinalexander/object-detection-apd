"""Map the device's own events onto the 17 Sep recordings, using the recorder index.

Inputs
  data/recording/17/db         recorder index: exact start/end time per clip
  data/recording/17/event.db   device events (to 18 Sep 02:41)
  data/prod_out/snapshots/     device snapshots, only for the events copied earlier

An event's timestamp is processing time, so it lands after the filmed frame. For
events whose device snapshot we hold, the frame is found by picture (NCC of the
person box over a window). Those matches give the lag distribution, and every
other event is placed at its timestamp minus the median lag.

Output, same layout as the replay arms:
  artifacts/replay/prod_device_17/apd_storage/apd_events.db      (camera_id = clip)
  artifacts/replay/prod_device_17/apd_storage/snapshots/<clip>/<event>.jpg
  artifacts/replay/prod_device_17/alignment.csv, summary.md

    python scripts/126_map_device_17.py
"""
import csv, datetime as dt, json, shutil, sqlite3, statistics, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
REC, IDX, EVDB = ROOT/"data/recording/17", ROOT/"data/recording/17/db", ROOT/"data/recording/17/event.db"
SNAP = ROOT/"data/prod_out/snapshots"
OUT = ROOT/"artifacts/replay/prod_device_17"
LAG_WINDOW = 45          # seconds before the event timestamp to search
P = lambda s: dt.datetime.fromisoformat(s) if s else None

idx = sqlite3.connect(f"file:{IDX}?mode=ro", uri=True)
clips = {}
for n, s, e, d in idx.execute("select file_name,start_time,end_time,duration_seconds from recordings"):
    if (REC/n).exists():
        clips[n] = (P(s), P(e) or P(s) + dt.timedelta(seconds=d or 0))
win = sorted((s, e, n) for n, (s, e) in clips.items())
print(f"{len(clips)} recordings indexed, {win[0][0]} .. {win[-1][1]}")

ev = list(sqlite3.connect(f"file:{EVDB}?mode=ro", uri=True).execute(
    "select event_id,timestamp,device_id,camera_id,session_id,track_id,person_bbox,violation,image_path from events"))
cols = ["event_id", "timestamp", "device_id", "camera_id", "session_id", "track_id", "person_bbox", "violation", "image_path"]
ev = [dict(zip(cols, r)) for r in ev]


def clip_of(t, lag=40):
    for s, e, n in win:
        if s - dt.timedelta(seconds=2) <= t <= e + dt.timedelta(seconds=lag): return n


for r in ev: r["t"] = P(r["timestamp"]); r["clip"] = clip_of(r["t"])
inside = [r for r in ev if r["clip"]]
print(f"{len(ev)} events, {len(inside)} inside a recording, {len({r['clip'] for r in inside})} clips")

# ---------------------------------------------------------------- picture match
def ncc(a, b):
    a = cv2.cvtColor(cv2.resize(a, (64, 128)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(b, (64, 128)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    a -= a.mean(); b -= b.mean()
    return float((a*b).sum()/max(1e-6, np.sqrt((a*a).sum()*(b*b).sum())))


def find_frame(clip, box, ref, t_end):
    """Best matching frame in [t_end-LAG_WINDOW, t_end], sampled at 5 fps."""
    cap = cv2.VideoCapture(str(REC/clip)); best = (None, -2)
    start = clips[clip][0]
    lo = max(0.0, (t_end - dt.timedelta(seconds=LAG_WINDOW) - start).total_seconds())
    hi = max(lo, (t_end - start).total_seconds())
    x1, y1, x2, y2 = box
    for off in np.arange(lo, hi + 0.2, 0.2):
        cap.set(cv2.CAP_PROP_POS_MSEC, off*1000); ok, f = cap.read()
        if not ok: continue
        c = f[max(0, y1):y2, max(0, x1):x2]
        if c.size == 0 or ref.size == 0: continue
        s = ncc(ref, c)
        if s > best[1]: best = (off, s)
    cap.release()
    return best


rows, lags = [], []
for r in inside:
    box = [int(v) for v in json.loads(r["person_bbox"])]
    sp = SNAP/Path(r["image_path"]).parent.name/Path(r["image_path"]).name
    if not sp.exists(): sp = SNAP/"masak"/Path(r["image_path"]).name
    r["snap"] = sp if sp.exists() else None
    if r["snap"] is not None:
        im = cv2.imread(str(r["snap"]))
        ref = im[max(0, box[1]):box[3], max(0, box[0]):box[2]] if im is not None else np.zeros((0, 0, 3), np.uint8)
        off, score = find_frame(r["clip"], box, ref, r["t"])
        if off is not None and score >= 0.75:
            r["offset"], r["score"], r["how"] = off, score, "picture"
            lags.append((r["t"] - (clips[r["clip"]][0] + dt.timedelta(seconds=off))).total_seconds())
            continue
    r["offset"], r["score"], r["how"] = None, None, "time"
med = statistics.median(lags) if lags else 9.6
print(f"picture-matched {len(lags)} events; device lag median {med:.1f}s "
      f"(range {min(lags):.1f}..{max(lags):.1f})" if lags else "no picture matches; assuming 9.6s lag")
for r in inside:
    if r["offset"] is None:
        r["offset"] = max(0.0, (r["t"] - clips[r["clip"]][0]).total_seconds() - med)

# ---------------------------------------------------------------- write
if OUT.exists(): shutil.rmtree(OUT)
(OUT/"apd_storage/snapshots").mkdir(parents=True)
db = sqlite3.connect(OUT/"apd_storage/apd_events.db")
db.executescript("""
create table events (id integer primary key autoincrement, event_id text unique, timestamp text,
 device_id text, camera_id text, session_id text, track_id integer, person_bbox text, violation text,
 image_path text, sent integer default 0);
create table apd_detail (id integer primary key autoincrement, event_id text, ppe_type text, status text, bbox text);
""")
det = {}
for e, p, s, b in sqlite3.connect(f"file:{EVDB}?mode=ro", uri=True).execute(
        "select event_id,ppe_type,status,bbox from apd_detail"):
    det.setdefault(e, []).append((p, s, b))

for r in inside:
    clip = r["clip"]; d = OUT/"apd_storage/snapshots"/Path(clip).stem
    d.mkdir(exist_ok=True)
    dst = d/f"{r['event_id']}.jpg"
    if r["snap"] is not None:
        shutil.copy2(r["snap"], dst)                       # the device's own snapshot
    else:                                                   # reconstruct from the recording
        cap = cv2.VideoCapture(str(REC/clip)); cap.set(cv2.CAP_PROP_POS_MSEC, r["offset"]*1000)
        ok, f = cap.read(); cap.release()
        if ok:
            box = [int(v) for v in json.loads(r["person_bbox"])]
            cv2.rectangle(f, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(f, ",".join(json.loads(r["violation"])) + "  [rekonstruksi]", (box[0], max(14, box[1]-6)),
                        0, 0.5, (0, 0, 255), 2)
            cv2.imwrite(str(dst), f)
    filmed = clips[clip][0] + dt.timedelta(seconds=r["offset"])
    db.execute("insert into events (event_id,timestamp,device_id,camera_id,session_id,track_id,person_bbox,violation,image_path)"
               " values (?,?,?,?,?,?,?,?,?)",
               (r["event_id"], filmed.strftime("%Y-%m-%d %H:%M:%S"), r["device_id"], Path(clip).stem,
                r["session_id"], r["track_id"], r["person_bbox"], r["violation"],
                str(dst.relative_to(OUT))))
    for p, s, b in det.get(r["event_id"], []):
        db.execute("insert into apd_detail (event_id,ppe_type,status,bbox) values (?,?,?,?)", (r["event_id"], p, s, b))
    rows.append({"event_id": r["event_id"], "device_time": r["timestamp"], "clip": Path(clip).stem,
                 "clip_offset_s": round(r["offset"], 2), "lag_s": round((r["t"]-filmed).total_seconds(), 1),
                 "match": r["how"], "score": round(r["score"], 3) if r["score"] else "",
                 "snapshot": "device" if r["snap"] is not None else "reconstructed"})
db.commit()
with open(OUT/"alignment.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

import collections
v = collections.Counter()
for r in inside:
    for i in json.loads(r["violation"]): v[i] += 1
byc = collections.Counter(Path(r["clip"]).stem for r in inside)
(OUT/"summary.md").write_text(
    f"""# Device events mapped onto the 17 Sep recordings

source: `data/recording/17/event.db` (to 2026-09-18 02:41), clip windows from
`data/recording/17/db` (the recorder's own index, exact start/end per clip).

| | |
|---|---|
| events in the database | {len(ev)} |
| inside one of the {len(clips)} copied recordings | **{len(inside)}** |
| clips with at least one event | {len(byc)} |
| violations | {", ".join(f"{k} {n}" for k, n in sorted(v.items()))} |
| placed by picture | {len(lags)} |
| placed by timestamp minus median lag | {len(inside)-len(lags)} |
| device lag behind the filmed frame | median {med:.1f}s{f", range {min(lags):.1f}..{max(lags):.1f}" if lags else ""} |

Snapshots: the device's own image where it was copied
({sum(1 for r in inside if r['snap'] is not None)}), otherwise the frame from the
recording with the person box drawn, marked "rekonstruksi"
({sum(1 for r in inside if r['snap'] is None)}).

Timestamps in apd_events.db are FILMED time, so they line up with a replay of the
same clips.
""")
print(f"wrote {OUT}: {len(inside)} events, {len(byc)} clips, violations {dict(v)}")
