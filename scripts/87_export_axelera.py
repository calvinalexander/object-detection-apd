"""Export our classifiers in the shape the Axelera YAML pipeline can declare.

The packed two-input graph is the right object for the AxRuntime/AxInferenceNet
route, where the application feeds tensors directly. It cannot be expressed in
the YAML pipeline builder, which gives every model exactly one input
(`type: image`, or `source: roi`). So the trunks are also exported singly:

    apd_ours_head.onnx  [1,3,128,128] -> head_cap, mask   (one head ROI)
    apd_ours_hand.onnx  [1,3,128,128] -> gloves           (one hand ROI)

Batch 1, because a secondary model in a cascade is invoked per ROI and the
runtime handles batching itself.
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchvision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import assert_contract

assert_contract()
PX = 128
OUT = Path("runs/packed"); OUT.mkdir(parents=True, exist_ok=True)

def load_one(arch, path):
    net = getattr(torchvision.models, arch)(weights=None)
    if hasattr(net, "classifier") and isinstance(net.classifier, nn.Sequential):
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    else:
        net.fc = nn.Linear(net.fc.in_features, 1)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd)
    return net.eval()

class HeadNet(nn.Module):
    """Two independent trunks on the same head crop -> two verdicts."""
    def __init__(self, hc, mk):
        super().__init__(); self.head_cap_net, self.mask_net = hc, mk
    def forward(self, x):
        return torch.cat([torch.sigmoid(self.head_cap_net(x)),
                          torch.sigmoid(self.mask_net(x))], dim=1)

class HandNet(nn.Module):
    def __init__(self, gl):
        super().__init__(); self.gloves_net = gl
    def forward(self, x): return torch.sigmoid(self.gloves_net(x))

hc = load_one("resnet34", "runs/fixed_sdk/resnet34/head_cap_best.pt")
mk = load_one("efficientnet_b0", "runs/bakeoff_torch/efficientnet_b0/mask_best.pt")
gl = load_one("efficientnet_b0", "runs/bakeoff_torch/efficientnet_b0/gloves_best.pt")

import onnx
for name, net, outs in (("apd_ours_head", HeadNet(hc, mk).eval(), ["head_cap_mask"]),
                        ("apd_ours_hand", HandNet(gl).eval(), ["gloves"])):
    p = OUT / f"{name}.onnx"
    torch.onnx.export(net, torch.randn(1, 3, PX, PX), str(p),
                      input_names=["input"], output_names=outs, opset_version=17)
    onnx.save_model(onnx.load(str(p)), str(p), save_as_external_data=False)
    Path(str(p) + ".data").unlink(missing_ok=True)
    n = sum(x.numel() for x in net.parameters())/1e6
    print(f"  {name+'.onnx':24} [1,3,{PX},{PX}] -> {outs[0]:14} {n:5.1f}M  {p.stat().st_size/1e6:6.1f} MB")

# sanity: the single-input exports still equal the source models
import onnxruntime as ort
x = np.random.randn(1,3,PX,PX).astype(np.float32)
h = ort.InferenceSession(str(OUT/"apd_ours_head.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
d = ort.InferenceSession(str(OUT/"apd_ours_hand.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
with torch.no_grad():
    r = (float(torch.sigmoid(hc(torch.from_numpy(x)))[0,0]),
         float(torch.sigmoid(mk(torch.from_numpy(x)))[0,0]),
         float(torch.sigmoid(gl(torch.from_numpy(x)))[0,0]))
print(f"\n  max diff vs source models: {max(abs(h[0,0]-r[0]), abs(h[0,1]-r[1]), abs(d[0,0]-r[2])):.2e}")
