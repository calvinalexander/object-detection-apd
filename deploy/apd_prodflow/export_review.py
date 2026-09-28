"""Collect the person filter's events into one folder for review off the box.

    python3 export_review.py <tray_counter dir> <out dir> [--skip N]

Reads <tray_counter>/apd_storage/person_filter.jsonl from line N+1 on and writes

    <out>/index.csv              one row per event: decision, production vs classifier, six scores, files
    <out>/person_filter.jsonl    the same lines, verbatim
    <out>/<decision>/<ts>_<cam>_<production>-to-<classifier>_<id8>.jpg        full frame (if still on disk)
    <out>/<decision>/<ts>_<cam>_<production>-to-<classifier>_<id8>_crop.png   what the network judged

decision is kept / passed / unscored. Kept frames are production's snapshots,
which its retention deletes after upload, so older kept events may have only
the crop. Prints the number of lines read, for the next --skip.
"""
import argparse, csv, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--skip", type=int, default=0)
a = ap.parse_args()
root, out = Path(a.root), Path(a.out)
st = root/"apd_storage"
SH = {"headcap": "cap", "mask": "mask", "gloves": "glv"}
ITEMS = ("head_cap", "mask", "gloves")

lines = (st/"person_filter.jsonl").read_text().splitlines()
new = lines[a.skip:]
out.mkdir(parents=True, exist_ok=True)
rows, missing = [], 0
with open(out/"person_filter.jsonl", "w") as fh:
    fh.write("\n".join(new) + ("\n" if new else ""))
for line in new:
    try:
        r = json.loads(line)
    except ValueError:
        continue                                   # a line cut by a crash mid-write
    eid, cam, dec = r["event_id"], str(r.get("camera_id", "cam")), r.get("decision", "unscored")
    clf = [g for g, v in (r.get("classifier") or {}).items() if v == "violation"]
    stem = (r["ts"].replace(" ", "_").replace(":", "-") + f"_{cam}_"
            + ("+".join(SH.get(g, g) for g in r.get("production", [])) or "none") + "-to-"
            + ("+".join(SH.get(g, g) for g in clf) or "clear") + f"_{eid[:8]}")
    d = out/dec; d.mkdir(exist_ok=True)
    frame = next((p for p in (st/"snapshots"/cam/f"{eid}.jpg", st/"snapshots_passed"/cam/f"{eid}.jpg") if p.exists()), None)
    crop = r.get("crop_path")
    crop = (Path(crop) if Path(crop).is_absolute() else root/crop) if crop else None
    fr = cr = ""
    if frame is not None:
        shutil.copy2(frame, d/f"{stem}.jpg"); fr = f"{dec}/{stem}.jpg"
    if crop is not None and crop.exists():
        shutil.copy2(crop, d/f"{stem}_crop.png"); cr = f"{dec}/{stem}_crop.png"
    if not fr and not cr:
        missing += 1
    p = r.get("p") or [None]*6
    rows.append({"ts": r["ts"], "event_id": eid, "camera": cam, "track": r.get("track_id"), "mode": r.get("mode"),
                 "model": r.get("model"), "decision": dec, "production": "+".join(r.get("production", [])),
                 "classifier": "+".join(clf), **{f"ours_{g}": (r.get("classifier") or {}).get(g, "") for g in SH},
                 **{f"p_{i}": p[k] for k, i in enumerate(ITEMS)}, **{f"known_{i}": p[3+k] for k, i in enumerate(ITEMS)},
                 "frame": fr, "crop": cr})
if rows:
    with open(out/"index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
n = {k: sum(1 for x in rows if x["decision"] == k) for k in ("kept", "passed", "unscored")}
print(f"events {len(rows)}  {n}  without any image {missing}")
print(f"LINES {len(lines)}")
