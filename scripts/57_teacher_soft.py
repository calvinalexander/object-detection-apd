"""Record the teacher's probabilities, not its verdicts.

The first distillation trained on argmax verdicts, which throws away almost
everything distillation exists to transfer: how confident the teacher was, and
which cases sit near its boundary. It also discarded every person the teacher
abstained on -- 376 of them -- when a soft target is perfectly well defined
there and says something useful ("this one is genuinely ambiguous").

Both voters are recorded separately rather than averaged, so the student can be
trained against either, or against their mean, without recomputing.
"""
import argparse, json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from apd.schema import assert_contract

assert_contract()
ATTRS = ("head_cap", "mask", "gloves")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/distill_roi")
ap.add_argument("--out", default="data/distill_roi/soft.jsonl")
a = ap.parse_args()

root = Path(a.data)
rows = [json.loads(l) for l in (root / "index.jsonl").read_text().splitlines()]
dev = "cuda"
probes = pickle.load(open("artifacts/voter3_ci3/probes.pkl", "rb"))
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
enc = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
y26 = {k: YOLO(f"runs/classify/runs/classify/runs/ci3_{k}/weights/best.pt") for k in ATTRS}
y26i = {k: {v: i for i, v in m.names.items()} for k, m in y26.items()}

@torch.no_grad()
def embed(ims, b=64):
    out = []
    for i in range(0, len(ims), b):
        out.append(enc(**proc(images=ims[i:i+b], return_tensors="pt").to(dev))
                   .last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

out = []
for n, r in enumerate(rows, 1):
    soft = {}
    if r["head"]:
        im = cv2.imread(str(root / "head" / r["head"]))
        e = embed([Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))])[0]
        for at in ("head_cap", "mask"):
            soft[at] = {
                "v3": float(probes[at].predict_proba(e[None])[0, 1]),
                "v4": float(y26[at].predict(im, verbose=False, device=dev)[0]
                            .probs.data[y26i[at]["present"]]),
            }
    if r["hands"]:
        ims = [cv2.imread(str(root / "hand" / h)) for h in r["hands"][:2]]
        E = embed([Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in ims])
        p3 = [float(probes["gloves"].predict_proba(e[None])[0, 1]) for e in E]
        p4 = [float(y26["gloves"].predict(i, verbose=False, device=dev)[0]
                    .probs.data[y26i["gloves"]["present"]]) for i in ims]
        # Both hands must be gloved, so the person's probability is the weaker hand's.
        soft["gloves"] = {"v3": float(min(p3)), "v4": float(min(p4)),
                          "hands3": p3, "hands4": p4}
    out.append({"stem": r["stem"], "soft": soft})
    if n % 400 == 0:
        print(f"  {n}/{len(rows)}", flush=True)

Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in out))
print(f"\n{len(out)} records -> {a.out}")
for at in ATTRS:
    v = [x["soft"][at]["v3"] for x in out if at in x["soft"]]
    w = [x["soft"][at]["v4"] for x in out if at in x["soft"]]
    if not v: continue
    mid = sum(1 for p in v if 0.2 < p < 0.8) / len(v)
    print(f"  {at:9} n={len(v):5d}  voter3 mean {np.mean(v):.3f}  voter4 mean {np.mean(w):.3f}  "
          f"|v3-v4| {np.mean(np.abs(np.array(v)-np.array(w))):.3f}  "
          f"{mid:.1%} of teacher probs are mid-range (lost by argmax)")
