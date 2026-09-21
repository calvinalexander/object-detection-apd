"""Why ours gets gloves and mask wrong: every outcome on the human labels, saved as images.

Deployed configuration: yolo11n-pose at 0.40, packed trunks for head_cap/mask
(thresholds from APD_OURS_*_THRESH, mask 0.60), crop detector for gloves at 0.25. Each labelled person is sorted into one outcome and a
crop is written to artifacts/error_analysis/<attribute>/<outcome>/.
"""
import json, collections, sys, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"deploy/apd_ours"))
os.environ["APD_OURS_GLOVES_ONNX"] = "apd_ours_gloves.onnx"
from apd_perception_ours import OursPerception, _letterbox, DET_NAMES
from ultralytics import YOLO

OUT = ROOT/"artifacts/error_analysis"
clf = OursPerception(SimpleNamespace(backend="axelera", gate=1))
pose = YOLO(str(ROOT/"yolo11n-pose.pt"))
persons = {json.loads(l)["person_id"]: json.loads(l) for l in open(ROOT/"artifacts/persons.jsonl")}
truth = collections.defaultdict(dict)
for l in open(ROOT/"artifacts/seed_labels.jsonl"):
    r = json.loads(l)
    if r["person_id"] in persons:
        for at, v in (r.get("labels") or {}).items():
            if v in ("present", "absent"): truth[at][r["person_id"]] = v
GRP = {"head_cap": "headcap", "mask": "mask", "gloves": "gloves"}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix*iy
    return i / max(1e-6, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def raw_gloves(im, pb):
    """Everything the crop detector said about this person, above 0.05."""
    H, W = im.shape[:2]; x1, y1, x2, y2 = pb; pw, ph = x2-x1, y2-y1
    c = im[int(max(0, y1-.18*ph)):int(min(H, y2+.18*ph)), int(max(0, x1-.18*pw)):int(min(W, x2+.18*pw))]
    if c.size == 0: return []
    x, *_ = _letterbox(c, 416)
    o = clf.gloves_det.run(None, {clf.gloves_det.get_inputs()[0].name: x})[0][0]
    return sorted(((DET_NAMES[int(r[5])], float(r[4])) for r in o if r[4] >= .05), key=lambda t: -t[1])[:4]


def save(at, outcome, im, pb, extra, name):
    H, W = im.shape[:2]; x1, y1, x2, y2 = [int(v) for v in pb]
    pw, ph = x2-x1, y2-y1
    c = im[max(0, y1-ph//6):min(H, y2+ph//6), max(0, x1-pw//4):min(W, x2+pw//4)].copy()
    s = 260 / max(c.shape[:2]); c = cv2.resize(c, (max(1, int(c.shape[1]*s)), max(1, int(c.shape[0]*s))))
    bar = np.full((34 + 14*len(extra), max(c.shape[1], 260), 3), 25, np.uint8)
    cv2.putText(bar, f"{outcome}  person {pw}x{ph}px", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (230, 230, 230), 1)
    for i, t in enumerate(extra):
        cv2.putText(bar, t, (4, 30+14*i), cv2.FONT_HERSHEY_SIMPLEX, .38, (160, 220, 160), 1)
    pad = np.full((c.shape[0], bar.shape[1], 3), 25, np.uint8); pad[:, :c.shape[1]] = c
    d = OUT/at/outcome; d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d/f"{name}.jpg"), np.vstack([bar, pad]))


cm = {a: collections.Counter() for a in GRP}
sizes = {a: collections.defaultdict(list) for a in GRP}
for fn in sorted({persons[p]["frame"] for a in truth for p in truth[a]}):
    im = cv2.imread(str(ROOT/"data/sppg"/fn))
    if im is None: continue
    r = pose.predict(im, imgsz=640, conf=0.40, verbose=False)[0]
    b = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
    k = r.keypoints.data.cpu().numpy() if len(r.boxes) else np.zeros((0, 17, 3))
    pb, vd = clf.perceive(im, b, k)
    for at in GRP:
        for pid, tv in truth[at].items():
            rec = persons[pid]
            if rec["frame"] != fn: continue
            t = "violation" if tv == "absent" else "ok"
            best, bi = 0.0, -1
            for i, x in enumerate(pb):
                v = iou(x, rec["person_box"])
                if v > best: best, bi = v, i
            box = rec["person_box"]; h = box[3]-box[1]
            name = pid.replace("#", "_")[-40:]
            if bi < 0 or best < .3:
                out = "not_located"; extra = [f"truth {t}; pose found no match (best IoU {best:.2f})"]
            else:
                st = vd[bi]["apd"].get(GRP[at], "unknown"); cf = vd[bi]["conf"].get(GRP[at])
                out = ("abstained" if st == "unknown" else
                       f"correct_{t}" if st == t else f"WRONG_said_{st}_truth_{t}")
                extra = [f"truth {t}  predicted {st}" + (f" ({cf:.2f})" if cf else "")]
                if at == "gloves":
                    extra.append("detector saw: " + (", ".join(f"{n} {s:.2f}" for n, s in raw_gloves(im, pb[bi])) or "nothing above 0.05"))
            cm[at][out] += 1; sizes[at][out].append(h)
            save(at, out, im, box, extra, name)

# the previous run, mask threshold 0.5, for the before/after table
BEFORE = {"mask": {"not_located": 25, "correct_violation": 13, "WRONG_said_ok_truth_violation": 4, "correct_ok": 2},
          "head_cap": {"not_located": 104, "correct_ok": 42, "correct_violation": 34,
                       "WRONG_said_ok_truth_violation": 14, "WRONG_said_violation_truth_ok": 7, "abstained": 1}}

lines = ["# Error analysis -- deployed configuration on human labels", "",
         f"yolo11n-pose at 0.40. Packed trunks for head_cap and mask, 'ok' when p(present) >= "
         f"{clf.thresh['headcap']} for head_cap and {clf.thresh['mask']} for mask. "
         f"Crop detector for gloves at {clf.det_conf}.",
         "Every labelled person is sorted into one outcome; a crop of each is in the folder of the same name.", ""]
for at in GRP:
    tot = sum(cm[at].values())
    lines += [f"## {at} -- {tot} labelled people", ""]
    if at in BEFORE:
        keys = sorted(set(BEFORE[at]) | set(cm[at]), key=lambda k: -max(BEFORE[at].get(k, 0), cm[at].get(k, 0)))
        lines += [f"| outcome | mask thresh 0.5 | now ({clf.thresh[GRP[at]]}) | median person height |", "|---|---|---|---|"]
        for o in keys:
            h = f"{np.median(sizes[at][o]):.0f} px" if sizes[at][o] else "-"
            lines.append(f"| `{at}/{o}/` | {BEFORE[at].get(o, 0)} | {cm[at].get(o, 0)} | {h} |")
    else:
        lines += ["| outcome | count | share | median person height |", "|---|---|---|---|"]
        for o, v in cm[at].most_common():
            lines.append(f"| `{at}/{o}/` | {v} | {v/tot:.0%} | {np.median(sizes[at][o]):.0f} px |")
    lines.append("")
lines += [(ROOT/"scripts/99_error_analysis_notes.md").read_text()]
(OUT/"README.md").write_text("\n".join(lines))
print("\n".join(lines[:60]))
