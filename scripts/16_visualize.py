"""Render end-to-end detections on held-out frames.

Draws what the pipeline actually produces: pose-derived person boxes, the head and
hand ROIs the classifiers see, and a per-attribute verdict. Frames come from
test_unseen_scene -- camera views no part of this was trained on.
"""
import json, sys, collections, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel
from apd.pose import frontality
from apd.schema import Attribute, Value

ap = argparse.ArgumentParser()
ap.add_argument("--n-frames", type=int, default=6)
ap.add_argument("--out", default="artifacts/detections.png")
args = ap.parse_args()

CROPS = Path("artifacts/crops"); ROOT = Path("data/sppg")
DEVICE, MODEL_ID = "cuda", "facebook/dinov2-base"
OK, BAD, UNK = (46, 160, 108), (214, 69, 55), (150, 158, 162)

persons = {p["person_id"]: p for p in (json.loads(l) for l in
           Path("artifacts/persons.jsonl").read_text().splitlines())}
index = [json.loads(l) for l in Path("artifacts/crops_index.jsonl").read_text().splitlines()]
head_crop = {r["person_id"]: r["crop"] for r in index if r["part"] == "head"}
hand_crop = collections.defaultdict(list)
for r in index:
    if r["part"] == "hand":
        hand_crop[r["person_id"]].append(r["crop"])
splits = json.loads(Path("artifacts/splits.json").read_text())["splits"]
fs = {f: k for k, v in splits.items() for f in v}

def gated(pid):
    p = persons[pid]
    return (p["head_roi"] and not p["head_collision"]
            and p["person_conf"] >= 0.35 and frontality(p["kp"]) >= 0.5)

proc = AutoImageProcessor.from_pretrained(MODEL_ID)
enc = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

@torch.no_grad()
def embed(paths, batch=96):
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(CROPS / p).convert("RGB") for p in paths[i:i + batch]]
        out.append(enc(**proc(images=ims, return_tensors="pt").to(DEVICE)
                       ).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out)

# ---- training sets, train split only -------------------------------------
seed = {r["person_id"]: r for r in (json.loads(l) for l in
        Path("artifacts/seed_labels.jsonl").read_text().splitlines())}
mined = [m for m in (json.loads(l) for l in
         Path("artifacts/mined_labels.jsonl").read_text().splitlines()) if m["attribute"] == "mask"]

def train_head(attr, extra=()):
    rows = []
    for pid, r in seed.items():
        v = r["labels"][attr]
        if v == Value.UNCLEAR.value or pid not in head_crop: continue
        if fs.get(r["frame"]) != "train": continue
        if attr == "mask" and not gated(pid): continue
        rows.append((head_crop[pid], int(v == Value.PRESENT.value)))
    for m in extra:
        if (m["value"] != "unclear" and m["split"] == "train"
                and m["person_id"] in head_crop and gated(m["person_id"])):
            rows.append((head_crop[m["person_id"]], int(m["value"] == "present")))
    seen, ded = set(), []
    for c, y in rows:
        if c not in seen: seen.add(c); ded.append((c, y))
    X = embed([c for c, _ in ded]); y = np.array([v for _, v in ded])
    return LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(X, y), len(y), int(y.sum())

def train_hand():
    rows = []
    for pid, r in seed.items():
        if fs.get(r["frame"]) != "train": continue
        for h in r.get("hands", []):
            crops = hand_crop.get(pid, [])
            if h["hand"] < len(crops) and h["value"] != Value.UNCLEAR.value:
                rows.append((crops[h["hand"]], int(h["value"] == Value.PRESENT.value)))
    X = embed([c for c, _ in rows]); y = np.array([v for _, v in rows])
    return LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(X, y), len(y), int(y.sum())

clf_cap, n_cap, p_cap = train_head(Attribute.HEAD_CAP.value)
clf_mask, n_mask, p_mask = train_head(Attribute.MASK.value, extra=mined)
clf_glove, n_gl, p_gl = train_hand()
print(f"trained on train split only -- head_cap {p_cap}/{n_cap}, "
      f"mask {p_mask}/{n_mask}, hand-gloved {p_gl}/{n_gl}")

# ---- pick busy held-out frames -------------------------------------------
by_frame = collections.defaultdict(list)
for pid, p in persons.items():
    if fs.get(p["frame"]) == "test_unseen_scene" and p["person_conf"] >= 0.5:
        by_frame[p["frame"]].append(pid)
cand = sorted(by_frame.items(), key=lambda kv: -len(kv[1]))
seen_scene, frames = set(), []
for f, pids in cand:
    sc = persons[pids[0]]["scene_id"]
    if sc in seen_scene: continue
    seen_scene.add(sc); frames.append((f, pids))
    if len(frames) >= args.n_frames: break

def verdict(clf, crop):
    pr = float(clf.predict_proba(embed([crop]))[0, 1])
    return ("present" if pr >= 0.5 else "absent"), pr

W = 900
panels = []
for fname, pids in frames:
    im = Image.open(ROOT / fname).convert("RGB")
    sc = W / im.width
    im = im.resize((W, int(im.height * sc)), Image.LANCZOS)
    dr = ImageDraw.Draw(im, "RGBA")
    for pid in pids:
        p = persons[pid]
        x1, y1, x2, y2 = [v * sc for v in p["person_box"]]
        res = {}
        if p["head_roi"] and pid in head_crop and not p["head_collision"]:
            res["cap"] = verdict(clf_cap, head_crop[pid])
            res["mask"] = verdict(clf_mask, head_crop[pid]) if gated(pid) else ("unclear", None)
        else:
            res["cap"] = res["mask"] = ("unclear", None)
        hands = hand_crop.get(pid, [])
        if len(hands) < 2:
            res["gloves"] = ("unclear", None)
        else:
            vs = [verdict(clf_glove, c) for c in hands[:2]]
            res["gloves"] = ("present" if all(v == "present" for v, _ in vs) else "absent",
                             min(pr for _, pr in vs))
        states = [v for v, _ in res.values()]
        colour = BAD if "absent" in states else (OK if "present" in states else UNK)
        dr.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        if p["head_roi"]:
            hx1, hy1, hx2, hy2 = [v * sc for v in p["head_roi"]]
            dr.rectangle([hx1, hy1, hx2, hy2], outline=colour + (140,), width=1)
        lines = []
        for k in ("cap", "mask", "gloves"):
            v, pr = res[k]
            mark = {"present": "OK", "absent": "X", "unclear": "?"}[v]
            lines.append(f"{mark} {k}" + (f" {pr:.2f}" if pr is not None else ""))
        th = 13 * len(lines) + 6
        ty = max(0, y1 - th - 2)
        dr.rectangle([x1, ty, x1 + 96, ty + th], fill=(17, 23, 26, 225))
        for i, t in enumerate(lines):
            v = res[("cap", "mask", "gloves")[i]][0]
            dr.text((x1 + 4, ty + 3 + 13 * i), t,
                    fill=OK if v == "present" else (BAD if v == "absent" else UNK))
    band = Image.new("RGB", (W, 22), (17, 23, 26))
    ImageDraw.Draw(band).text((6, 5), f"{fname[:52]}   scene {persons[pids[0]]['scene_id']}"
                              f"   {len(pids)} persons   [test_unseen_scene]", fill=(230, 236, 238))
    panel = Image.new("RGB", (W, im.height + 22), (233, 237, 238))
    panel.paste(band, (0, 0)); panel.paste(im, (0, 22))
    panels.append(panel)

total_h = sum(p.height + 8 for p in panels) + 30
sheet = Image.new("RGB", (W, total_h), (233, 237, 238))
d = ImageDraw.Draw(sheet)
d.text((6, 8), "OK = compliant   X = violation   ? = abstain (not judgeable)   "
               "box red if any violation", fill=(40, 50, 54))
y = 26
for p in panels:
    sheet.paste(p, (0, y)); y += p.height + 8
sheet.save(args.out)
print(f"wrote {args.out}  ({len(panels)} frames)")
