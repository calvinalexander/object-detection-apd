"""Label the cleaned site crops with three independent views; accept unanimity.

Measured against the human COCO labels (scripts/45), no single view dominates:
the ROI sheet is cheap and abstains least on heads, the person crop is the most
accurate on heads, and the three views abstain on DIFFERENT people -- which is
why their union decides 23% of gloves where the ROI sheet alone decides 7%.

A crop decided by at least one view and contradicted by none is accepted. A
contradiction becomes `unclear`: two views looking at the same person and
disagreeing is evidence the evidence was not there.

Every frame is cleaned of the dashboard's burned-in verdict first, and crops
overlapping a caption were already rejected at mining time.
"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.labeller import label_sheet, render_sheet
from apd.labeller2 import label_panels, render_panels, _fit
from apd.overlay import clean
from apd.schema import Attribute, assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--attribute", default=Attribute.HEAD_CAP.value)
ap.add_argument("--model", default="opus")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--root", default="artifacts/site_crops_clean")
args = ap.parse_args()
attr = args.attribute
root = Path(args.root) / attr
sel = [json.loads(l) for l in (root / "selection.jsonl").read_text().splitlines()]
work = root / "views"; work.mkdir(exist_ok=True)
print(f"{attr}: {len(sel)} crops, 3 views each")

_frames: dict[str, np.ndarray] = {}
def frame(path):
    if path not in _frames:
        _frames[path] = clean(cv2.imread(path))
    return _frames[path]

def view_a(i, grp):                       # ROI contact sheet, 12 per sheet
    sheet = work / f"a{i:03d}.jpg"
    render_sheet([root / "pool" / r["crop"] for r in grp], sheet)
    return label_sheet(sheet, len(grp), attr, model=args.model)

def view_b(i, grp):                       # whole frame + region, 3 per sheet
    items = [{"frame": frame(r["image"]), "person_box": r["person_box"],
              "roi_box": r["box"]} for r in grp]
    sheet = work / f"b{i:03d}.jpg"
    render_panels(items, sheet)
    return label_panels(sheet, len(grp), attr, model=args.model)

def view_c(i, grp):                       # person crop enlarged, 5 per sheet
    cells = []
    for r in grp:
        x1, y1, x2, y2 = r["person_box"]
        f = frame(r["image"]); pad = int(0.12 * max(x2 - x1, y2 - y1))
        crop = f[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        cells.append(_fit(cv2.resize(crop if crop.size else f, None, fx=3, fy=3,
                                     interpolation=cv2.INTER_CUBIC), 300, 460))
    sheet_img = np.full((486, 300 * len(cells), 3), 236, np.uint8)
    sheet_img[:26] = (19, 26, 29)
    for n, c in enumerate(cells):
        cv2.putText(sheet_img, f"{n+1}", (n * 300 + 8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        sheet_img[26:, n * 300:(n + 1) * 300] = c
    p = work / f"c{i:03d}.jpg"
    cv2.imwrite(str(p), sheet_img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return label_sheet(p, len(grp), attr, model=args.model)

VIEWS = [("a", view_a, 12), ("b", view_b, 3), ("c", view_c, 5)]
votes: dict[str, dict[str, str]] = {r["crop"]: {} for r in sel}
cost = 0.0
t0 = time.time()
for tag, fn, per in VIEWS:
    groups = [sel[i:i + per] for i in range(0, len(sel), per)]
    done = errs = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for grp, res in ex.map(lambda ig: (ig[1], fn(ig[0], ig[1])), enumerate(groups)):
            cost += res.cost_usd or 0.0
            if res.error:
                errs += 1
            else:
                for n, r in enumerate(grp):
                    v = res.verdicts.get(str(n + 1))
                    if v:
                        votes[r["crop"]][tag] = v
            done += 1
    dec = sum(1 for v in votes.values() if v.get(tag) and v[tag] != "unclear")
    print(f"  view {tag}: {done} sheets, decided {dec}/{len(sel)} "
          f"({dec/len(sel):.1%})  ${cost:.2f}  {time.time()-t0:.0f}s  errors={errs}")

rows, tally, conflicts = [], {}, 0
for r in sel:
    vs = [v for v in votes[r["crop"]].values() if v != "unclear"]
    if len(set(vs)) > 1:
        value, conflicts = "unclear", conflicts + 1
    else:
        value = vs[0] if vs else "unclear"
    tally[value] = tally.get(value, 0) + 1
    rows.append(r | {"value": value, "votes": votes[r["crop"]], "labeller": args.model})
(root / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"\n{attr}: {tally}  conflicts dropped {conflicts}  ${cost:.2f}")
