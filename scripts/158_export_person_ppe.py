"""Export a person-PPE checkpoint to the deployable ONNX + JSON pair.

Same contract as the v3 file the device already runs: input `person_crops`
[n,3,320,320] (letterboxed, ImageNet-normalised RGB), output `p` [n,6] =
sigmoid(logits) in the order present_{head_cap,mask,gloves}, known_{...}.
The cuts written to the JSON are the ones fitted on the manual curation
(scripts/148), not the val cuts in summary.json.

    python scripts/158_export_person_ppe.py --run runs/person_ppe_v5/efficientnet_b0_320 --name v5 \
        --out deploy/apd_prodflow
"""
import argparse, json
from pathlib import Path
import numpy as np, onnxruntime as ort, torch, torchvision

ROOT = Path(__file__).resolve().parents[1]
V5_CUTS = {"head_cap": {"known_above": 0.95, "present_below": 0.05},
           "mask": {"known_above": 0.10, "present_below": 0.275},
           "gloves": {"known_above": 0.80, "present_below": 0.10}}
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/person_ppe_v5/efficientnet_b0_320")
ap.add_argument("--name", default="v5")
ap.add_argument("--out", default="deploy/apd_prodflow")
ap.add_argument("--cuts", default=json.dumps(V5_CUTS), help="JSON of per-item known_above/present_below")
a = ap.parse_args()

j = json.loads((ROOT/a.run/"summary.json").read_text())
net = getattr(torchvision.models, j["arch"])(weights=None)
net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 6)
net.load_state_dict(torch.load(ROOT/a.run/"best.pt", map_location="cpu", weights_only=False))
model = torch.nn.Sequential(net, torch.nn.Sigmoid()).eval()

out = ROOT/a.out; out.mkdir(parents=True, exist_ok=True)
onnx_path = out/f"apd_ours_person_ppe_{a.name}.onnx"
x = torch.randn(2, 3, j["px"], j["px"])
torch.onnx.export(model, x, str(onnx_path), input_names=["person_crops"], output_names=["p"],
                  dynamic_axes={"person_crops": {0: "n"}, "p": {0: "n"}}, opset_version=17, dynamo=False)

xs = np.random.RandomState(0).randn(5, 3, j["px"], j["px"]).astype(np.float32) * 0.5
ref = model(torch.from_numpy(xs)).detach().numpy()
got = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, {"person_crops": xs})[0]
err = float(np.abs(ref - got).max())
assert err < 1e-4, f"onnx differs from torch by {err}"

meta = {
    "input": "person crop = person box + 18% pad, letterbox 320 (pad 114), BGR->RGB, /255, ImageNet mean/std, NCHW",
    "outputs": j["outputs"],
    "rule": "judge only when p_known >= known_above; then violation when p_present < present_below; else unknown",
    "cuts": json.loads(a.cuts),
    "cuts_val": j["cuts"],
    "calibrated_on": "manual curation of production's 16 Sep events, even hours (scripts/148); held out on odd hours",
    "checkpoint": str(Path(a.run)/"best.pt"), "best_epoch": j["best_epoch"],
}
onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(f"wrote {onnx_path} (+ .json), max |onnx - torch| = {err:.2e}")
