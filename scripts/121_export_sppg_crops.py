"""Export the reviewed sppg people (data/label_sppg/labels.csv) as head and face crops
for the head_cap / mask classifiers, on the device's geometry, split by scene (15% val).

    data/clean/sppg_head_cap/{train,val}/{present,absent}/<id>.jpg   head_roi -> _crop 128
    data/clean/sppg_mask/{train,val}/{present,absent}/<id>.jpg       face_roi 0.8 -> _crop 128
unclear / not_person are skipped. No gate is applied: a reviewer judged the item visible.

    python scripts/121_export_sppg_crops.py
"""
import collections, csv, hashlib, json, shutil, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
from apd.pose import head_roi, face_roi
from apd.schema import GEOMETRY
from apd_perception_ours import _crop

P = {json.loads(l)["person_id"].replace("#", "_p"): json.loads(l) for l in open(ROOT/"artifacts/persons.jsonl")}
out = {"head_cap": ROOT/"data/clean/sppg_head_cap", "mask": ROOT/"data/clean/sppg_mask"}
for d in out.values():
    if d.exists(): shutil.rmtree(d)
n = collections.Counter(); cache = {}
for r in csv.DictReader(open(ROOT/"data/label_sppg/labels.csv")):
    sp = "val" if int(hashlib.md5(("s" + r["scene"]).encode()).hexdigest(), 16) % 100 < 15 else "train"
    p = P[r["id"]]; kp = np.array(p["kp"], float); box = tuple(p["person_box"])
    if r["frame"] not in cache: cache = {r["frame"]: cv2.imread(str(ROOT/"data/sppg"/r["frame"]))}
    im = cache[r["frame"]]
    for it, roi in (("head_cap", lambda: head_roi(kp, box)), ("mask", lambda: face_roi(kp, box, gain=0.8))):
        if r[it] not in ("present", "absent"): continue
        b = roi()
        c = None if b is None else _crop(im, b, GEOMETRY.head_crop_px)
        if c is None: n[(it, sp, "no_roi")] += 1; continue
        d = out[it]/sp/r[it]; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d/f"{r['id']}.jpg"), c); n[(it, sp, r[it])] += 1
for k in sorted(n): print(k, n[k])
