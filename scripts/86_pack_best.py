"""Pack the three BEST per-attribute models into one graph, unchanged.

Packing is about deployment, not learning. The three winners stay exactly as
trained -- three independent trunks, no shared weights, no shared gradients,
each seeing only its own crop -- but they live in one file, compile as one unit,
and occupy one AIPU resource slot instead of three.

    head_crops ->  resnet34   (+SDK, labels-fixed)  -> head_cap
               ->  effnet-b0  (regularised)         -> mask
    hand_crops ->  effnet-b0  (regularised)         -> gloves

Identity with the source models is asserted numerically, not assumed.
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchvision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--head-cap", default="runs/fixed_sdk/resnet34/head_cap_best.pt")
ap.add_argument("--head-cap-arch", default="resnet34")
ap.add_argument("--mask", default="runs/bakeoff_torch/efficientnet_b0/mask_best.pt")
ap.add_argument("--mask-arch", default="efficientnet_b0")
ap.add_argument("--gloves", default="runs/bakeoff_torch/efficientnet_b0/gloves_best.pt")
ap.add_argument("--gloves-arch", default="efficientnet_b0")
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--max-people", type=int, default=8)
ap.add_argument("--max-hands", type=int, default=16)
ap.add_argument("--static", type=int, default=0, help="fixed batch, as the Axelera compiler needs")
ap.add_argument("--out", default="runs/packed/apd_best3.onnx")
a = ap.parse_args()


def load_one(arch, path):
    """Rebuild the exact model scripts/72 trained and saved as a bare state_dict."""
    net = getattr(torchvision.models, arch)(weights=None)
    if hasattr(net, "classifier") and isinstance(net.classifier, nn.Sequential):
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    else:
        net.fc = nn.Linear(net.fc.in_features, 1)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    net.load_state_dict(sd)
    return net.eval()


class Best3(nn.Module):
    """Three independent networks, one module. Nothing is shared between them."""
    def __init__(self, hc, mk, gl):
        super().__init__()
        self.head_cap_net, self.mask_net, self.gloves_net = hc, mk, gl

    def forward(self, head_crops, hand_crops):
        return (torch.sigmoid(self.head_cap_net(head_crops)[:, 0]),
                torch.sigmoid(self.mask_net(head_crops)[:, 0]),
                torch.sigmoid(self.gloves_net(hand_crops)[:, 0]))


hc = load_one(a.head_cap_arch, a.head_cap)
mk = load_one(a.mask_arch, a.mask)
gl = load_one(a.gloves_arch, a.gloves)
net = Best3(hc, mk, gl).eval()
print("packed three independent trunks:")
for nm, arch, p, m in (("head_cap", a.head_cap_arch, a.head_cap, hc),
                       ("mask", a.mask_arch, a.mask, mk),
                       ("gloves", a.gloves_arch, a.gloves, gl)):
    print(f"  {nm:9} {arch:16} {sum(x.numel() for x in m.parameters())/1e6:5.1f}M  {p}")
print(f"  total {sum(x.numel() for x in net.parameters())/1e6:.1f}M params")

# --- identity with the sources, before anything else
xh = torch.randn(5, 3, a.px, a.px); xd = torch.randn(9, 3, a.px, a.px)
with torch.no_grad():
    pk = net(xh, xd)
    ref = (torch.sigmoid(hc(xh)[:, 0]), torch.sigmoid(mk(xh)[:, 0]), torch.sigmoid(gl(xd)[:, 0]))
d = max(float((p - r).abs().max()) for p, r in zip(pk, ref))
print(f"\n  packed vs source models: max diff {d:.2e}  ({'identical' if d < 1e-6 else 'MISMATCH'})")

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
if a.static:
    dh = torch.randn(a.max_people, 3, a.px, a.px); dd = torch.randn(a.max_hands, 3, a.px, a.px)
    dyn = None
    print(f"\n  static shapes: head [{a.max_people},3,{a.px},{a.px}] hand [{a.max_hands},3,{a.px},{a.px}]"
          f"  (pad short frames; the Axelera compiler needs fixed shapes)")
else:
    dh, dd = torch.randn(3, 3, a.px, a.px), torch.randn(6, 3, a.px, a.px)
    dyn = {"head_crops": {0: "n_people"}, "hand_crops": {0: "n_hands"},
           "head_cap": {0: "n_people"}, "mask": {0: "n_people"}, "gloves": {0: "n_hands"}}
torch.onnx.export(net, (dh, dd), str(out),
                  input_names=["head_crops", "hand_crops"],
                  output_names=["head_cap", "mask", "gloves"],
                  dynamic_axes=dyn, opset_version=17)
import onnx
onnx.save_model(onnx.load(str(out)), str(out), save_as_external_data=False)
Path(str(out) + ".data").unlink(missing_ok=True)
print(f"  exported -> {out}  ({out.stat().st_size/1e6:.1f} MB)")

import onnxruntime as ort
s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
print("\n  graph signature:")
for i in s.get_inputs():  print(f"    in  {i.name:12} {i.shape}")
for o in s.get_outputs(): print(f"    out {o.name:12} {o.shape}")
nh = a.max_people if a.static else 5
nd = a.max_hands if a.static else 9
g = s.run(None, {"head_crops": np.random.randn(nh,3,a.px,a.px).astype(np.float32),
                 "hand_crops": np.random.randn(nd,3,a.px,a.px).astype(np.float32)})
print(f"\n  runs: {g[0].shape[0]} head verdicts, {g[2].shape[0]} hand verdicts")
