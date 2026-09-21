"""Rank mask candidates so a labelling pass finds positives instead of blanks.

Random sampling is the wrong tool here: mask positives are rare, so most crops a
labeller sees would be negatives already well represented in the seed set. Instead
embed every eligible crop, score it with a weak classifier fit on the 18 known
positives, and label the top of the ranking first.

One rule this pass must not break: the ranking is used for TRAIN only.
Enriching val or test for positives would destroy their prevalence and make every
downstream precision number meaningless, so those are sampled at random.
"""
import json, sys, collections, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel
from apd.schema import Attribute, Value

CROPS = Path("artifacts/crops")
DEVICE, MODEL_ID = "cuda", "facebook/dinov2-base"
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--mode", choices=["ranked", "random"], default="ranked")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--out", default="artifacts/mine_sheet.png")
ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--per-scene", type=int, default=1,
                help="max crops per scene; 1 maximises information per label")
args = ap.parse_args()

if args.split != "train" and args.mode == "ranked":
    raise SystemExit("refusing to rank a held-out split: it would distort prevalence")

splits = json.loads(Path("artifacts/splits.json").read_text())["splits"]
frame_split = {f: k for k, v in splits.items() for f in v}
seed = {r["person_id"]: r for r in (json.loads(l) for l in
        Path("artifacts/seed_labels.jsonl").read_text().splitlines())}
index = [json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]

# Eligible: head crop, facing the camera, no neighbour's head in the ROI.
elig = [r for r in index if r["part"] == "head" and r.get("frontal_face")
        and not r.get("head_collision") and frame_split.get(r["frame"]) == args.split]
labelled = {p for p, r in seed.items() if r["labels"][Attribute.MASK.value] != Value.UNCLEAR.value}
todo = [r for r in elig if r["person_id"] not in labelled]
print(f"{args.split}: {len(elig)} eligible head crops, {len(todo)} unlabelled for mask")

proc = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

@torch.no_grad()
def embed(paths, batch=96):
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(CROPS / p).convert("RGB") for p in paths[i:i + batch]]
        px = proc(images=ims, return_tensors="pt").to(DEVICE)
        out.append(model(**px).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out)

if args.mode == "ranked":
    known = [(r["crop"], int(seed[r["person_id"]]["labels"][Attribute.MASK.value] == "present"))
             for r in index if r["part"] == "head" and r["person_id"] in labelled]
    Xk = embed([c for c, _ in known]); yk = np.array([y for _, y in known])
    print(f"weak scorer fit on {yk.sum()} positives / {len(yk)-yk.sum()} negatives")
    clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(Xk, yk)
    Xt = embed([r["crop"] for r in todo])
    prob = clf.predict_proba(Xt)[:, 1]
    order = np.argsort(-prob)
    # Score alone collapses onto one camera: the first pass returned 22 positives
    # spanning 3 scenes, 20 of them from one. Cap per scene so each label buys a
    # distinct worker and background rather than another frame of the same shift.
    seen, picked, scores = collections.Counter(), [], []
    for i in order:
        sc = todo[i]["scene_id"]
        if seen[sc] >= args.per_scene:
            continue
        seen[sc] += 1
        picked.append(todo[i]); scores.append(prob[i])
        if len(picked) >= args.skip + args.n:
            break
    picked, scores = picked[args.skip:], scores[args.skip:]
else:
    rng = np.random.default_rng(17)
    idx = rng.permutation(len(todo))[args.skip:args.skip + args.n]
    picked = [todo[i] for i in idx]; scores = [None] * len(picked)

CELL, COLS = 300, 4
ROWS = (len(picked) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * CELL, ROWS * (CELL + 24)), (233, 237, 238))
dr = ImageDraw.Draw(sheet)
for n, r in enumerate(picked):
    im = Image.open(CROPS / r["crop"]).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    x, y = (n % COLS) * CELL, (n // COLS) * (CELL + 24)
    sheet.paste(im, (x, y + 24))
    dr.rectangle([x, y, x + CELL - 1, y + 23], fill=(19, 26, 29))
    tag = f"M{n+1:02d}" + (f"  p={scores[n]:.2f}" if scores[n] is not None else "")
    dr.text((x + 7, y + 6), tag, fill=(255, 255, 255))
sheet.save(args.out)
Path(args.out).with_suffix(".json").write_text(json.dumps(
    [{"slot": n + 1, "person_id": r["person_id"], "crop": r["crop"],
      "frame": r["frame"], "split": args.split} for n, r in enumerate(picked)], indent=1))
print(f"wrote {args.out} ({len(picked)} crops, mode={args.mode})")
