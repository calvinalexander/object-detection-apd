"""Export the 53 COCO-annotated frames as person crops + labels for training.

Until now these were the held-out test set. The user has decided to evaluate on
the recordings manually instead, so they become training data. Their "unclear"
labels are kept: they are the only judgeability supervision from this source.

    python scripts/132_export_coco_people.py
Writes data/clean/person_coco/{labels.csv, person/<id>.jpg}
"""
import collections, csv, json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"data/clean/person_coco"; (OUT/"person").mkdir(parents=True, exist_ok=True)
ITEMS = ["head_cap", "mask", "gloves"]
persons = {json.loads(l)["person_id"]: json.loads(l) for l in open(ROOT/"artifacts/persons.jsonl")}
lab = collections.defaultdict(dict)
for l in open(ROOT/"artifacts/seed_labels.jsonl"):
    r = json.loads(l)
    if r["person_id"] in persons:
        for at, v in (r.get("labels") or {}).items():
            if v in ("present", "absent", "unclear"): lab[r["person_id"]][at] = v
rows, n = [], collections.Counter()
for pid, items in sorted(lab.items()):
    rec = persons[pid]
    im = cv2.imread(str(ROOT/"data/sppg"/rec["frame"]))
    if im is None: continue
    H, W = im.shape[:2]
    x1, y1, x2, y2 = rec["person_box"]; pw, ph = x2-x1, y2-y1
    c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
    if c.size == 0 or min(c.shape[:2]) < 16: continue
    pid_s = pid.replace("#", "_p").replace("/", "_")
    cv2.imwrite(str(OUT/"person"/f"{pid_s}.jpg"), c)
    row = {"id": pid_s, "frame": rec["frame"], "scene": rec["scene_id"],
           "person": str((OUT/"person"/f"{pid_s}.jpg").relative_to(ROOT))}
    for it in ITEMS:
        row[it] = items.get(it, "")
        n[(it, row[it] or "no label")] += 1
    rows.append(row)
with open(OUT/"labels.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} COCO people -> {OUT}")
for k in sorted(n): print("  ", k, n[k])
