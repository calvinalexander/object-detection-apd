"""Contact sheets of the crops the probes are actually fitted on.

One panel per attribute, sampled the way a batch would be drawn: train split
only, balanced across present/absent, with the label and its provenance printed
on each tile. Provenance matters here -- seed labels come from human boxes, mined
and bulk labels were produced by a model with no human check this iteration, so
being able to eyeball them is the only quality control there is.
"""
import argparse, collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from apd.schema import Value

ap = argparse.ArgumentParser()
ap.add_argument("--attribute", default="head_cap", choices=["head_cap","mask","gloves"])
ap.add_argument("--n", type=int, default=48); ap.add_argument("--cols", type=int, default=8)
ap.add_argument("--cell", type=int, default=150)
ap.add_argument("--out", default=None)
ap.add_argument("--split", default="train")
args = ap.parse_args()
CROPS = Path("artifacts/crops")
OKC, BADC = (46,160,108), (206,64,52)

splits = json.loads(Path("artifacts/splits.json").read_text())["splits"]
fs = {f: k for k, v in splits.items() for f in v}
idx = [json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]
head = {r["person_id"]: r["crop"] for r in idx if r["part"] == "head"}
hand = {(r["person_id"], int(r["crop"].rsplit("_h",1)[1].split(".")[0])): r["crop"]
        for r in idx if r["part"] == "hand"}

rows = []   # (crop, value, source)
for r in (json.loads(l) for l in Path("artifacts/seed_labels.jsonl").read_text().splitlines()):
    if fs.get(r["frame"]) != args.split: continue
    if args.attribute == "gloves":
        for h in r.get("hands", []):
            c = hand.get((r["person_id"], h["hand"]))
            if c and h["value"] != Value.UNCLEAR.value: rows.append((c, h["value"], "seed"))
    else:
        v = r["labels"][args.attribute]
        if v != Value.UNCLEAR.value and r["person_id"] in head:
            rows.append((head[r["person_id"]], v, "seed"))
if args.attribute == "mask":
    for f, tag in (("mined_labels.jsonl","mined"), ("bulk_labels.jsonl","bulk")):
        p = Path("artifacts")/f
        if not p.exists(): continue
        for m in (json.loads(l) for l in p.read_text().splitlines()):
            if m.get("attribute") != "mask" or m["value"] == "unclear": continue
            if m.get("split") != args.split or m["person_id"] not in head: continue
            rows.append((head[m["person_id"]], m["value"], tag))

seen, ded = set(), []
for c, v, s in rows:
    if c not in seen: seen.add(c); ded.append((c, v, s))
pos = [r for r in ded if r[1] == "present"]; neg = [r for r in ded if r[1] == "absent"]
rng = np.random.default_rng(0); rng.shuffle(pos); rng.shuffle(neg)
half = args.n // 2
pick = pos[:half] + neg[:args.n - half]
rng.shuffle(pick)

src = collections.Counter(s for _, _, s in ded)
print(f"{args.attribute} [{args.split}]: {len(pos)} present / {len(neg)} absent   sources {dict(src)}")

C = args.cell; COLS = args.cols; ROWS = (len(pick)+COLS-1)//COLS
sheet = Image.new("RGB", (COLS*C, ROWS*(C+18)+24), (235,238,239))
d = ImageDraw.Draw(sheet)
d.text((6,7), f"{args.attribute}  [{args.split}]  {len(pos)} present / {len(neg)} absent"
              f"   -- showing {len(pick)}", fill=(30,38,42))
for i,(c,v,s) in enumerate(pick):
    im = Image.open(CROPS/c).convert("RGB").resize((C-4,C-4), Image.LANCZOS)
    x,y = (i%COLS)*C, 24+(i//COLS)*(C+18)
    col = OKC if v=="present" else BADC
    d.rectangle([x,y,x+C-1,y+C+16], fill=col)
    sheet.paste(im,(x+2,y+2))
    d.text((x+4,y+C), f"{'PRESENT' if v=='present' else 'ABSENT'}  {s}", fill=(255,255,255))
out = args.out or f"artifacts/train_batch_{args.attribute}.png"
sheet.save(out); print("wrote", out)
