"""Materialise part crops to disk, one decode per frame.

Crops are the training input, and they are small -- the whole set fits in page
cache, which matters because this box has 46 GB of GPU against only 4 vCPU, so a
live crop-on-load dataloader would starve the GPU.
"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from apd.schema import GEOMETRY

ROOT = Path("data/sppg")
OUT = Path("artifacts/crops")
PAD = 0.35   # context around the ROI; body orientation disambiguates a covered head

persons = [json.loads(l) for l in Path("artifacts/persons.jsonl").read_text().splitlines()]
by_frame = collections.defaultdict(list)
for p in persons:
    by_frame[p["frame"]].append(p)

(OUT / "head").mkdir(parents=True, exist_ok=True)
(OUT / "hand").mkdir(parents=True, exist_ok=True)
index, stats = [], collections.Counter()

def cut(im, box, size, path):
    x1, y1, x2, y2 = box
    pad = PAD * max(x2 - x1, y2 - y1)
    crop = im.crop((max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                    min(im.width, int(x2 + pad)), min(im.height, int(y2 + pad))))
    if crop.width < 8 or crop.height < 8:
        return False
    crop.resize((size, size), Image.LANCZOS).save(path, quality=92)
    return True

for n, (frame, people) in enumerate(sorted(by_frame.items()), 1):
    with Image.open(ROOT / frame) as im:
        im = im.convert("RGB")
        for p in people:
            pid = p["person_id"].replace("#", "_")
            if p["head_roi"] and cut(im, p["head_roi"], GEOMETRY.head_crop_px,
                                     OUT / "head" / f"{pid}.jpg"):
                stats["head"] += 1
                index.append({"crop": f"head/{pid}.jpg", "part": "head",
                              "person_id": p["person_id"], "frame": frame,
                              "scene_id": p["scene_id"],
                              "head_localizable": p["head_localizable"],
                              "frontal_face": p["frontal_face"],
                              "head_collision": p["head_collision"]})
            for h, box in enumerate(p["hand_rois"]):
                if cut(im, box, GEOMETRY.hand_crop_px, OUT / "hand" / f"{pid}_h{h}.jpg"):
                    stats["hand"] += 1
                    index.append({"crop": f"hand/{pid}_h{h}.jpg", "part": "hand",
                                  "person_id": p["person_id"], "frame": frame,
                                  "scene_id": p["scene_id"]})
    if n % 400 == 0:
        print(f"  {n}/{len(by_frame)} frames", flush=True)

Path("artifacts/crops_index.jsonl").write_text(
    "".join(json.dumps(r) + "\n" for r in index))
print(f"\nhead crops: {stats['head']}   hand crops: {stats['hand']}")
print(f"wrote {OUT} and artifacts/crops_index.jsonl")
