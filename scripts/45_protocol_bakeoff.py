"""Does the context panel actually label better? Measured, not assumed.

Ground truth is the human COCO annotation on the 53 SPPG frames (seed_box),
the only labels here not produced by a model. Both protocols see the same
persons. Reported separately, because conflating them is how "55% agreement"
once got reported when there were zero disagreements:

  decided   -- fraction the protocol was willing to answer
  accuracy  -- of those, fraction matching the human label
"""
import argparse, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.labeller import label_sheet, render_sheet
from apd.labeller2 import label_panels, render_panels
from apd.schema import Attribute, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--attribute", default=Attribute.HEAD_CAP.value)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--model", default="opus")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()
attr = args.attribute
part = "head" if attr in (Attribute.HEAD_CAP.value, Attribute.MASK.value) else "hand"

persons = {json.loads(l)["person_id"]: json.loads(l)
           for l in Path("artifacts/persons.jsonl").read_text().splitlines()}
truth = {}
for line in Path("artifacts/seed_labels.jsonl").read_text().splitlines():
    r = json.loads(line)
    v = (r.get("labels") or {}).get(attr)
    if v in ("present", "absent") and r["person_id"] in persons:
        truth[r["person_id"]] = v

crops = {}
for line in Path("artifacts/crops_index.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["part"] == part:
        crops.setdefault(r["person_id"], []).append(r["crop"])

pool = [p for p in truth if p in crops]
random.Random(args.seed).shuffle(pool)
pool = pool[:args.n]
print(f"{attr}: {len(pool)} persons with a human label "
      f"({sum(truth[p]=='present' for p in pool)} present / "
      f"{sum(truth[p]=='absent' for p in pool)} absent)")

FRAMES = Path("data/sppg")
CROPS = Path("artifacts/crops")
work = Path("artifacts/bakeoff") / attr
work.mkdir(parents=True, exist_ok=True)

# ---- protocol A: the current 12-per-sheet crop contact sheet ----------
def run_a(i_grp):
    i, grp = i_grp
    sheet = work / f"a{i:03d}.jpg"
    render_sheet([CROPS / crops[p][0] for p in grp], sheet)
    res = label_sheet(sheet, len(grp), attr, model=args.model)
    return grp, res

# ---- protocol B: context panel, 3 rows per sheet ----------------------
def run_b(i_grp):
    i, grp = i_grp
    items = []
    for p in grp:
        rec = persons[p]
        frame = cv2.imread(str(FRAMES / rec["frame"]))
        roi = rec["head_roi"] if part == "head" else (rec["hand_rois"] or [rec["head_roi"]])[0]
        items.append({"frame": frame, "person_box": rec["person_box"], "roi_box": roi})
    sheet = work / f"b{i:03d}.jpg"
    render_panels(items, sheet)
    res = label_panels(sheet, len(grp), attr, model=args.model)
    return grp, res

# ---- protocol C: the whole person, enlarged, aspect preserved -------
# A 96px hand ROI abstains 88% of the time. A person is a bigger, more
# reliably localised object; if the hands are readable at all, they are
# readable there, and the posture says whether hands are even in view.
def run_c(i_grp):
    i, grp = i_grp
    from apd.labeller2 import _fit
    import numpy as np
    cells = []
    for p in grp:
        rec = persons[p]
        frame = cv2.imread(str(FRAMES / rec["frame"]))
        x1, y1, x2, y2 = [int(v) for v in rec["person_box"]]
        pad = int(0.12 * max(x2 - x1, y2 - y1))
        crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if crop.size == 0:
            crop = frame
        cells.append(_fit(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                          300, 460))
    band = 26
    sheet = np.full((460 + band, 300 * len(cells), 3), 236, np.uint8)
    sheet[:band] = (19, 26, 29)
    for n, c in enumerate(cells):
        cv2.putText(sheet, f"{n+1}", (n * 300 + 8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        sheet[band:, n * 300:(n + 1) * 300] = c
    out = work / f"c{i:03d}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])
    from apd.labeller import label_sheet as _ls
    return grp, _ls(out, len(grp), attr, model=args.model)


def score(name, fn, groups):
    t0, cost, dec, corr, err = time.time(), 0.0, 0, 0, 0
    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for grp, res in ex.map(fn, enumerate(groups)):
            cost += res.cost_usd or 0.0
            if res.error:
                err += 1
                continue
            for n, p in enumerate(grp):
                v = res.verdicts.get(str(n + 1))
                if v is None:
                    continue
                rows.append({"person_id": p, "pred": v, "truth": truth[p]})
                if v != "unclear":
                    dec += 1
                    corr += int(v == truth[p])
    (work / f"{name.split()[0]}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"  {name:24} decided {dec}/{len(rows)} ({dec/max(1,len(rows)):5.1%})  "
          f"accuracy {corr}/{dec} ({corr/max(1,dec):5.1%})  "
          f"${cost:.2f}  {time.time()-t0:.0f}s  errors={err}")
    return rows

print("\nprotocol comparison against human labels:")
score("A crop sheet (12/sheet)", run_a, [pool[i:i+12] for i in range(0, len(pool), 12)])
score("B context panel (3/sheet)", run_b, [pool[i:i+3] for i in range(0, len(pool), 3)])
score("C person crop (5/sheet)", run_c, [pool[i:i+5] for i in range(0, len(pool), 5)])
