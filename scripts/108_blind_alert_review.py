"""Blind review of the violation alerts both arms raised on the same recordings.

  ours          artifacts/replay/ours          (replay, event time = video time)
  prod_device   artifacts/replay/prod_device   (the device's own events, only the
                                                ones 107 placed in a clip by picture)

Only clips the ours replay has finished are used, so both arms watched the same
footage. Each alert becomes one tile cut from the RAW recording frame (neither
arm's drawing is visible), the tiles are shuffled, and the arm is kept in
key.csv. A reviewer then marks every claimed violation real / wrong / unclear in
labels.json; `--score` joins the two and prints alert precision per arm.

  python scripts/108_blind_alert_review.py          # build tiles + sheets
  python scripts/108_blind_alert_review.py --score  # after labels.json exists
"""
import csv, datetime as dt, json, random, re, sqlite3, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/"artifacts/replay"; OUT = R/"review"
VIDS = {p.stem: p for p in (ROOT/"data/recording").rglob("*.mp4")}


def events(arm):
    db = sqlite3.connect(R/arm/"apd_storage/apd_events.db")
    return db.execute("""select event_id, timestamp, camera_id, person_bbox, violation
                         from events where violation != '[]'""").fetchall()


def build():
    done = {Path(json.loads(l)["clip"]).stem for l in open(R/"ours/progress.jsonl") if json.loads(l)["ok"]}
    done = {c for c in done if re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", c)}   # device events are matched to dated clips only
    placed = {r["event_id"]: float(r["clip_offset_s"]) for r in csv.DictReader(open(R/"prod_device/alignment.csv"))
              if r["placed"] == "by picture"}
    items = []
    for eid, ts, cam, pb, vio in events("ours"):
        if cam not in done: continue
        t = (dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(cam, "%Y-%m-%d_%H-%M-%S")).total_seconds() + 0.5
        items.append(("ours", eid, cam, t, json.loads(pb), json.loads(vio)))
    for eid, ts, cam, pb, vio in events("prod_device"):
        if eid in placed and cam in done:
            items.append(("prod_device", eid, cam, placed[eid], json.loads(pb), json.loads(vio)))
    random.Random(7).shuffle(items)
    (OUT/"tiles").mkdir(parents=True, exist_ok=True)
    key, tiles = [], []
    for i, (arm, eid, cam, t, pb, vio) in enumerate(items):
        cap = cv2.VideoCapture(str(VIDS[cam])); cap.set(cv2.CAP_PROP_POS_MSEC, t*1000); ok, f = cap.read()
        if not ok: continue
        H, W = f.shape[:2]; x1, y1, x2, y2 = pb; pw, ph = x2-x1, y2-y1
        p = f[max(0, y1-6):min(H, y2+6), max(0, x1-6):min(W, x2+6)]
        cx, cy, s = (x1+x2)//2, (y1+y2)//2, int(max(pw, ph)*1.3)
        c = f[max(0, cy-s):min(H, cy+s), max(0, cx-s):min(W, cx+s)].copy()
        cv2.rectangle(c, (x1-max(0, cx-s), y1-max(0, cy-s)), (x2-max(0, cx-s), y2-max(0, cy-s)), (0, 255, 255), 1)
        h = 360
        p = cv2.resize(p, (int(h*p.shape[1]/p.shape[0]), h), interpolation=cv2.INTER_CUBIC)
        c = cv2.resize(c, (int(h*c.shape[1]/c.shape[0]), h))
        tile = np.hstack([p, c]); tile = cv2.copyMakeBorder(tile, 28, 0, 0, max(0, 480-tile.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))
        tile = tile[:, :480]
        tid = f"{len(key):03d}"
        cv2.putText(tile, f"{tid}  claims: {', '.join(vio)}", (4, 20), 0, 0.6, (0, 0, 200), 2)
        cv2.imwrite(str(OUT/f"tiles/{tid}.jpg"), tile)
        key.append(dict(tile=tid, arm=arm, event_id=eid, clip=cam, t=round(t, 1), claims=vio)); tiles.append(tile)
    with open(OUT/"key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    for pg in range(0, len(tiles), 8):
        t = tiles[pg:pg+8]; t += [np.full_like(tiles[0], 255)]*(8-len(t))
        cv2.imwrite(str(OUT/f"sheet_{pg//8:02d}.jpg"), np.vstack([np.hstack(t[j:j+4]) for j in range(0, 8, 4)]))
    n = {a: sum(k["arm"] == a for k in key) for a in ("ours", "prod_device")}
    print(f"{len(done)} clips, {len(key)} tiles {n}, sheets in {OUT}")


def score():
    key = list(csv.DictReader(open(OUT/"key.csv")))
    lab = json.loads((OUT/"labels.json").read_text())      # {"000": {"gloves": "real"|"wrong"|"unclear"}}
    print(f"{'arm':12} {'class':8} {'alerts':>6} {'real':>5} {'wrong':>5} {'unclear':>7} {'precision (clear)':>18}")
    for arm in ("ours", "prod_device"):
        for g in ("headcap", "mask", "gloves", "all"):
            c = {"real": 0, "wrong": 0, "unclear": 0}
            for k in key:
                if k["arm"] != arm: continue
                for cl in json.loads(k["claims"].replace("'", '"')):
                    if g in ("all", cl): c[lab[k["tile"]][cl]] += 1
            n = sum(c.values()); d = c["real"]+c["wrong"]
            print(f"{arm:12} {g:8} {n:6d} {c['real']:5d} {c['wrong']:5d} {c['unclear']:7d} {c['real']/d if d else float('nan'):18.1%}")


if __name__ == "__main__":
    score() if "--score" in sys.argv else build()
