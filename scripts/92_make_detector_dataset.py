"""Turn everything we have into one 7-class detector dataset, production's schema.

A detector needs frames with boxes. We hold three kinds of thing:

  1. frames already annotated as boxes      -> copy the labels through
  2. per-person attribute decisions with a  -> emit one box per decided attribute
     box recorded alongside (consensus)
  3. per-person attribute labels whose only -> join to persons.jsonl, which kept
     geometry is the crop it came from         person_box / head_roi / hand_rois

The dashboard caption is found exactly, not guessed: production paints it with
cv2.rectangle(..., (30,30,30), -1), so the fill colour is known and the rows of
that colour vote on the rectangle's edges.

The hard part is not the geometry, it is what to do with "unclear". A detector
learns from absence: an object present in the pixels but missing from the label
file is taught as background. Our unclear rate is 13% for head_cap but 74% for
mask, so emitting nothing there would train the detector to stay silent on
exactly the attribute it is worst at. Three policies are offered; `blank` is the
honest default for mask and gloves.

    drop    leave the region unlabelled     (teaches suppression -- use only where unclear is rare)
    blank   fill the ROI with grey          (the region carries no evidence either way)
    frame   drop the whole frame            (cleanest, costs 86% of consensus frames)
"""
import argparse, collections, json, re, shutil, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]
IDX = {n: i for i, n in enumerate(NAMES)}
POS = {"head_cap": "head_cap", "mask": "mask", "gloves": "gloves"}
NEG = {"head_cap": "no_head_cap", "mask": "no_mask", "gloves": "no_gloves"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/det7")
ap.add_argument("--mode", choices=["person", "frame"], default="person",
                help="person: one image per person, cut out of the frame. The caption sits elsewhere, "
                     "so a crop that does not touch it carries no leak and needs nothing painted over, "
                     "which is what the grey blocks were for.")
ap.add_argument("--pad", type=float, default=0.18, help="person mode: padding around the person box")
ap.add_argument("--person-from-pose", type=int, default=1,
                help="frame mode: complete the person class with yolo11x-pose. Consensus person boxes "
                     "come from the production detector and miss people; a person present in the pixels "
                     "but absent from the label file is taught as background.")
ap.add_argument("--pose-weights", default="yolo11x-pose.pt")
ap.add_argument("--pose-conf", type=float, default=0.35)
ap.add_argument("--unclear", choices=["drop", "blank", "frame"], default="blank")
ap.add_argument("--boxes", choices=["annotated", "roi", "prefer-annotated"], default="annotated",
                help="annotated: only boxes the production model drew, which are tight. The ROI "
                     "fallbacks are anatomical estimates -- a hand ROI is 2.4x the hand and lands as a "
                     "blob over the torso -- so they fit badly and a detector trained on them learns "
                     "loose boxes. Cost: 3262 of 6186 site attribute boxes, gloves worst hit.")
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--sources", nargs="+", default=["native", "consensus", "persons"])
ap.add_argument("--pose-rois", type=int, default=1,
                help="consensus records no box for an unclear attribute; recover one with pose so it can be greyed out")
ap.add_argument("--clean-overlay", type=int, default=1,
                help="production snapshots carry the dashboard verdict caption; the label comes from that "
                     "same verdict, so leaving it in lets a detector read the answer off the image")
ap.add_argument("--need-attr", type=int, default=1,
                help="person mode: keep only crops that carry at least one attribute box. A crop with "
                     "just a person box gives no attribute supervision, and consensus person boxes are "
                     "model output, so some of them are a gas cylinder or a table.")
ap.add_argument("--drop-empty", type=int, default=1,
                help="a frame whose boxes were all dropped still shows people; YOLO would read it as "
                     "pure background and learn to suppress them")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
if a.mode == "person":
    a.unclear, a.pose_rois = "drop", 0        # nothing is painted over, so nothing needs a box

OUT = ROOT / a.out
COCO_FRAMES = {json.loads(l)["frame"] for l in open(ROOT/"artifacts/seed_labels.jsonl")}   # eval only, never train
print(f"{len(COCO_FRAMES)} COCO-annotated frames held out of training entirely\n")

# ------------------------------------------------- ROI recovery for unclears
_pose = {"model": None, "cache": {}}
def rois_for(frame_path, person_box):
    """(head_roi, hand_rois) for the pose person matching this consensus box."""
    if not a.pose_rois: return None, []
    key = str(frame_path)
    if key not in _pose["cache"]:
        if _pose["model"] is None:
            from ultralytics import YOLO
            from apd.pose import build_rois as _br
            _pose["model"] = YOLO(str(ROOT/"yolo11n-pose.pt")); _pose["build"] = _br
        im = cv2.imread(str(frame_path))
        if im is None: _pose["cache"][key] = []
        else:
            r = _pose["model"].predict(im, imgsz=960, conf=0.25, verbose=False)[0]
            if r.keypoints is None or not len(r.boxes): _pose["cache"][key] = []
            else:
                _pose["cache"][key] = _pose["build"](r.boxes.xyxy.cpu().numpy(),
                                                     r.keypoints.data.cpu().numpy())
    best, bi = 0.0, None
    for pp in _pose["cache"][key]:
        b = pp.person_box
        ix = max(0, min(b[2], person_box[2]) - max(b[0], person_box[0]))
        iy = max(0, min(b[3], person_box[3]) - max(b[1], person_box[1]))
        inter = ix*iy
        u = (b[2]-b[0])*(b[3]-b[1]) + (person_box[2]-person_box[0])*(person_box[3]-person_box[1]) - inter
        s_ = inter/max(1e-6, u)
        if s_ > best: best, bi = s_, pp
    if bi is None or best < 0.3:
        # no pose person: a head box from the person box, which is enough to grey out
        x1, y1, x2, y2 = person_box; h = (y2-y1)*0.28; w = (x2-x1)*0.55; cx = (x1+x2)/2
        return [cx-w/2, y1, cx+w/2, y1+h], []
    return (list(bi.head) if bi.head is not None else None), [list(h) for h in (bi.hands or [])]

# ---------------------------------------------------------------- sources
records = {}          # frame path -> {"boxes": [(cls, x1,y1,x2,y2)], "blank": [roi], "src": str}

def group_key(frame_path):
    """What must not straddle the split. Frames from one video or one site are
    near-duplicates of each other -- 57% of adjacent frames were within one
    frame of another in the earlier leak audit."""
    p = Path(frame_path); stem = p.stem
    if re.fullmatch(r"\d{2}\.\d{2}\.\d{2}\.\d{4}\.\d{2}", p.parent.name):
        return f"site:{p.parent.name}"                      # consensus frames live under their site
    m = re.match(r"([0-9a-f]{8})-", stem)
    if m: return f"vid:{m.group(1)}"                        # 8-hex video id from the SDK exports
    m = re.match(r"(\d{4}-\d{2}-\d{2}_\d+)", stem)
    if m: return f"scene:{m.group(1)}"
    return f"frame:{stem}"

def add(frame_path, cls, box, src, person=None):
    r = records.setdefault(str(frame_path), {"boxes": [], "blank": [], "src": src,
                                             "group": group_key(frame_path), "persons": []})
    r["boxes"].append((cls, *[float(v) for v in box]))
    if person is not None: r["persons"].append([float(v) for v in person])

def blank(frame_path, box, src):
    r = records.setdefault(str(frame_path), {"boxes": [], "blank": [], "src": src,
                                             "group": group_key(frame_path), "persons": []})
    r["blank"].append([float(v) for v in box])

# 1 -- frames already in detector format
NATIVE = [ROOT/"artifacts/sdk/sdk/customers/apd_v3/dataset", ROOT/"data/voter3",
          ROOT/"data/dump/dataset/apd_v2/dataset", ROOT/"data/dump/dataset/apd_v3/dataset"]
if "native" in a.sources:
    seen = set()
    for root in NATIVE:
        for split in ("train", "val", "test"):
            d = root/"images"/split
            if not d.exists(): continue
            for p in sorted(d.iterdir()):
                if p.name in seen or p.name in COCO_FRAMES: continue
                lp = root/"labels"/split/(p.stem+".txt")
                if not lp.exists(): continue
                im = cv2.imread(str(p))
                if im is None: continue
                H, W = im.shape[:2]; seen.add(p.name)
                for line in lp.read_text().splitlines():
                    f = line.split()
                    if len(f) < 5: continue
                    c, cx, cy, bw, bh = int(f[0]), *[float(v) for v in f[1:5]]
                    bx = ((cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H)
                    add(p, c, bx, "native", person=bx if c == IDX["person"] else None)
    print(f"native      {len(seen):5d} frames already boxed")

# 2 -- consensus: person box + a box per decided attribute
if "consensus" in a.sources:
    n_f = n_b = n_u = n_r = n_k = 0
    for line in open(ROOT/"artifacts/consensus.jsonl"):
        c = json.loads(line)
        fp = ROOT/"data/sites"/c["site"]/f'{c["id"]}.jpg'
        if not fp.exists() or fp.name in COCO_FRAMES: continue
        keep = True; pend = []
        for pp in c["persons"]:
            pend.append(("person", IDX["person"], pp["box"]))
            for at, v in pp.get("attrs", {}).items():
                dec, bx = v.get("decision"), v.get("box")
                if dec in ("present", "absent") and bx:
                    if a.boxes == "roi" and v.get("box_src") != "roi_fallback": continue
                    if a.boxes == "annotated" and v.get("box_src") != "apd_mbg": continue
                    pend.append(("box", IDX[POS[at] if dec == "present" else NEG[at]], bx)); n_b += 1
                elif dec == "unclear":
                    n_u += 1
                    if a.unclear == "frame":
                        keep = False
                    elif a.unclear == "blank":
                        if bx:
                            pend.append(("blank", None, bx))
                        else:
                            hr, hands = rois_for(fp, pp["box"])
                            if at == "gloves":
                                for h in hands: pend.append(("blank", None, h)); n_r += 1
                            elif hr: pend.append(("blank", None, hr)); n_r += 1
        if not keep: continue
        n_f += 1
        # head_cap and mask read the same head ROI. Mask is unclear 74% of the
        # time, so greying its ROI would erase the evidence for a head_cap that
        # WAS decided. A blank only survives if no kept box needs those pixels.
        kept = [bx for kind, cls, bx in pend if kind == "box" and cls != IDX["person"]]
        def _needed(b):
            for k in kept:
                ix = max(0, min(b[2], k[2]) - max(b[0], k[0])); iy = max(0, min(b[3], k[3]) - max(b[1], k[1]))
                if ix*iy / max(1.0, (k[2]-k[0])*(k[3]-k[1])) > 0.25: return True
            return False
        for kind, cls, bx in pend:
            if kind == "person": add(fp, cls, bx, "consensus", person=bx)
            elif kind == "box": add(fp, cls, bx, "consensus")
            elif not _needed(bx): blank(fp, bx, "consensus"); n_k += 1
    print(f"consensus   {n_f:5d} frames, {n_b} attribute boxes, {n_u} unclear "
          f"({n_r} ROIs recovered by pose, {n_k} greyed after sparing those a kept box needs)")

# 3 -- crop-folder labels joined to persons.jsonl geometry
if "persons" in a.sources:
    persons = {}
    for l in open(ROOT/"artifacts/persons.jsonl"):
        d = json.loads(l); persons[d["person_id"]] = d
    def base(name):
        s = re.sub(r"__[a-z0-9]+\.jpg$", "", name); s = re.sub(r"\.jpg$", "", s)
        m = re.match(r"(.+?)_(\d+)(?:_h\d+)?$", s)
        return f"{m.group(1)}#{m.group(2)}" if m else None
    lab = collections.defaultdict(dict)
    for at in ("head_cap", "mask", "gloves"):
        for split in ("train", "val"):
            for cls in ("present", "absent"):
                d = ROOT/"data/crops_ci_inpaint"/at/split/cls
                if not d.exists(): continue
                for f in d.iterdir():
                    pid = base(f.name)
                    if pid in persons: lab[pid][at] = cls
    frames = ROOT/"data/sppg"
    n_p = n_b = 0
    for pid, attrs in lab.items():
        d = persons[pid]
        fp = frames/d["frame"]
        if not fp.exists() or fp.name in COCO_FRAMES: continue
        n_p += 1
        add(fp, IDX["person"], d["person_box"], "persons")
        for at, cls in attrs.items():
            cid = IDX[POS[at] if cls == "present" else NEG[at]]
            if at == "gloves":
                for hr in (d.get("hand_rois") or []): add(fp, cid, hr, "persons"); n_b += 1
            elif d.get("head_roi"):
                add(fp, cid, d["head_roi"], "persons"); n_b += 1
    print(f"persons     {n_p:5d} persons joined to geometry, {n_b} attribute boxes")

print(f"\ntotal frames {len(records)}")
cnt = collections.Counter(c for r in records.values() for c, *_ in r["boxes"])
print("boxes: " + "  ".join(f"{NAMES[i]}={cnt[i]}" for i in range(7) if cnt[i]))
blanks = sum(len(r["blank"]) for r in records.values())
if a.unclear == "blank": print(f"regions to grey out: {blanks}")
if a.dry_run: sys.exit(0)

# ---------------------------------------------------------------- write
def _extend(m, x1, y1, x2, y2, fw, fh, frac=0.5, limit=48, patience=14):
    """A text line spanning the panel breaks its rows, so the vote stops short
    of the real edge. Grow while the next line is still mostly panel colour,
    tolerating one text line before giving up, and trim back to the last line
    that passed. This cannot run away the way a brightness test does: the test
    is the exact fill colour, and the true edge drops to near zero and stays.
    """
    def sweep(v, stop, step, line):
        best, miss = v, 0
        while v != stop and abs(v-best) < limit and miss <= patience:
            v += step
            i = v if step > 0 else v-1
            if not (0 <= i < stop or 0 <= i): break
            try: ok = line(i) >= frac
            except IndexError: break
            if ok: best, miss = v, 0
            else: miss += 1
        return best
    y2 = sweep(y2, fh, +1, lambda i: m[i, x1:x2].mean())
    y1 = sweep(y1, 0, -1, lambda i: m[i, x1:x2].mean())
    x2 = sweep(x2, fw, +1, lambda i: m[y1:y2, i].mean())
    x1 = sweep(x1, 0, -1, lambda i: m[y1:y2, i].mean())
    return x1, y1, x2, y2


def panel_rects(im, tol=12, min_w=40, min_h=24, min_rows=6):
    """Exactly the rectangle draw_table paints: cv2.rectangle(..., (30,30,30), -1).

    Identification rather than estimation, because the fill colour is fixed in
    production's own code. Connected components fail here: the panel touches
    dark scenery and the bounding box then spans both. A filled rectangle has a
    stronger signature, many rows sharing one left and right edge, so the rows
    vote on the edges and the text glyphs punching holes do not matter.
    """
    fh, fw = im.shape[:2]
    m = (np.abs(im.astype(np.int16) - 30).max(axis=2) <= tol)
    runs = collections.defaultdict(list)
    for y in range(fh):
        row = m[y]
        if not row.any(): continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if e - s >= min_w: runs[(s//3, e//3)].append((y, s, e))
    out = []
    for rs in runs.values():
        if len(rs) < min_rows: continue
        ys = [r[0] for r in rs]; y1, y2 = min(ys), max(ys) + 1
        if y2 - y1 < min_h: continue
        x1, x2 = min(r[1] for r in rs), max(r[2] for r in rs)
        if (x2-x1) > 0.5*fw or (y2-y1) > 0.45*fh: continue
        patch = im[y1:y2, x1:x2].reshape(-1, 3)
        if abs(float(np.median(patch)) - 30) > 6: continue
        if (np.abs(patch.astype(np.int16)-30).max(axis=1) <= tol).mean() < 0.45: continue
        x1, y1, x2, y2 = _extend(m, x1, y1, x2, y2, fw, fh)
        out.append((int(x1), int(y1), int(x2), int(y2)))
    out.sort(key=lambda r: -(r[2]-r[0])*(r[3]-r[1]))
    keep = []
    for r in out:
        if not any(r[0]>=k[0]-2 and r[1]>=k[1]-2 and r[2]<=k[2]+2 and r[3]<=k[3]+2 for k in keep):
            keep.append(r)
    return keep


def _covered(box, rects, frac=0.25):
    x1, y1, x2, y2 = box; area = max(1.0, (x2-x1)*(y2-y1))
    for r in rects:
        ix = max(0, min(x2, r[2]) - max(x1, r[0])); iy = max(0, min(y2, r[3]) - max(y1, r[1]))
        if ix*iy/area >= frac: return True
    return False
groups = collections.defaultdict(list)
for f, r in records.items(): groups[r["group"]].append(f)
keys = sorted(groups); rng = np.random.default_rng(20260916); rng.shuffle(keys)
n_val = max(1, int(len(keys)*a.val_frac)); val_keys = set(keys[:n_val])
print(f"\n{len(keys)} groups -> {len(val_keys)} val groups")

for split in ("train", "val"):
    (OUT/"images"/split).mkdir(parents=True, exist_ok=True)
    (OUT/"labels"/split).mkdir(parents=True, exist_ok=True)
w = collections.Counter()
def _write(split, name, im, boxes):
    """One image plus its YOLO label file, boxes given in this image's pixels."""
    H, W = im.shape[:2]
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, y1 = max(0.0, x1), max(0.0, y1); x2, y2 = min(float(W), x2), min(float(H), y2)
        if x2 - x1 < 2 or y2 - y1 < 2: continue
        lines.append(f"{c} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    if not lines and a.drop_empty:
        w["empty_dropped"] += 1; return False
    cv2.imwrite(str(OUT/"images"/split/name), im)
    (OUT/"labels"/split/f"{Path(name).stem}.txt").write_text("\n".join(lines))
    w[split] += 1
    return True


def _inside(box, region, frac=0.6):
    """Is this attribute box mostly inside this person's crop?"""
    ix = max(0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0, min(box[3], region[3]) - max(box[1], region[1]))
    return ix*iy / max(1.0, (box[2]-box[0])*(box[3]-box[1])) >= frac


_pose_m = {"m": None}
def pose_persons(im):
    if _pose_m["m"] is None:
        from ultralytics import YOLO
        _pose_m["m"] = YOLO(str(ROOT/a.pose_weights))
    r = _pose_m["m"].predict(im, imgsz=960, conf=a.pose_conf, verbose=False)[0]
    return [] if not len(r.boxes) else [tuple(float(v) for v in b) for b in r.boxes.xyxy.cpu().numpy()]


def _iou(x, y):
    ix = max(0, min(x[2], y[2]) - max(x[0], y[0])); iy = max(0, min(x[3], y[3]) - max(x[1], y[1]))
    inter = ix*iy
    return inter / max(1e-6, (x[2]-x[0])*(x[3]-x[1]) + (y[2]-y[0])*(y[3]-y[1]) - inter)


def _hits(box, rects):
    for r in rects:
        if (min(box[2], r[2]) - max(box[0], r[0])) > 0 and (min(box[3], r[3]) - max(box[1], r[1])) > 0:
            return True
    return False


for g, files in groups.items():
    split = "val" if g in val_keys else "train"
    for f in files:
        r = records[f]; src = Path(f)
        im = cv2.imread(str(src))
        if im is None: continue
        H, W = im.shape[:2]
        # The exact rectangle production paints, found by its fill colour.
        # Crop mode skips any person it touches; frame mode paints it out.
        rects = panel_rects(im) if r["src"] == "consensus" else []

        if a.mode == "frame" and a.person_from_pose:
            # Annotator person boxes stay as they are; pose only adds people
            # nobody boxed. Same teacher that labelled the ROIs, so the boxes
            # are consistent with the rest of the pipeline.
            for pb in pose_persons(im):
                if not any(_iou(pb, q) > 0.45 for q in r["persons"]):
                    r["boxes"].append((IDX["person"], *pb)); r["persons"].append(list(pb))
                    w["persons_added_by_pose"] += 1

        if a.mode == "person":
            # One image per person, cut out of the frame. The caption sits
            # elsewhere, so a crop that does not touch it carries no leak and
            # needs no painting over -- which is what the greyed blocks were for.
            attrs = [t for t in r["boxes"] if t[0] != IDX["person"]]
            for i, pb in enumerate(r["persons"]):
                if _hits(pb, rects):
                    w["persons_touching_caption"] += 1; continue
                pw, ph = pb[2]-pb[0], pb[3]-pb[1]
                px, py = a.pad*pw, a.pad*ph
                cx1, cy1 = int(max(0, pb[0]-px)), int(max(0, pb[1]-py))
                cx2, cy2 = int(min(W, pb[2]+px)), int(min(H, pb[3]+py))
                if cx2-cx1 < 32 or cy2-cy1 < 48:
                    w["persons_too_small"] += 1; continue
                crop = im[cy1:cy2, cx1:cx2].copy()
                mine = [(c, x1-cx1, y1-cy1, x2-cx1, y2-cy1)
                        for c, x1, y1, x2, y2 in attrs if _inside((x1, y1, x2, y2), (cx1, cy1, cx2, cy2))]
                if not mine and a.need_attr:
                    w["persons_no_attr"] += 1; continue
                mine.append((IDX["person"], pb[0]-cx1, pb[1]-cy1, pb[2]-cx1, pb[3]-cy1))
                _write(split, f"{src.stem}_p{i}.jpg", crop, mine)
                w["persons"] += 1
            if rects: w["cleaned"] += 1
            continue

        if rects:
            for x1, y1, x2, y2 in rects: im[y1:y2, x1:x2] = 114
            before = len(r["boxes"])
            r["boxes"] = [t for t in r["boxes"] if not _covered(t[1:], rects)]
            w["boxes_dropped"] += before - len(r["boxes"]); w["cleaned"] += 1
        for x1, y1, x2, y2 in r["blank"]:
            x1, y1 = max(0, int(x1)), max(0, int(y1)); x2, y2 = min(W, int(x2)), min(H, int(y2))
            if x2 > x1 and y2 > y1: im[y1:y2, x1:x2] = 114
        _write(split, f"{src.stem}.jpg", im, r["boxes"])
(OUT/"dataset.yaml").write_text(
    f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: 7\nnames:\n" +
    "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
print(f"wrote {w['train']} train / {w['val']} val images to {OUT}")
if a.mode == "frame" and a.person_from_pose:
    print(f"pose added {w['persons_added_by_pose']} person boxes nobody had labelled")
if a.mode == "person":
    print(f"{w['persons']} person crops; skipped {w['persons_touching_caption']} touching the caption, "
          f"{w['persons_no_attr']} with no attribute box, {w['persons_too_small']} too small to read")
if a.clean_overlay and a.mode == "frame":
    print(f"dashboard caption removed from {w['cleaned']} snapshots; "
          f"{w['boxes_dropped']} boxes dropped because the caption covered them")
if a.drop_empty: print(f"{w['empty_dropped']} frames dropped for having no box left")
