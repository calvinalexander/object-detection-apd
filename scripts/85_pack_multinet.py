"""Package the separate trunks as ONE network with multiple inputs.

"Separate models" and "a single architecture" are not in conflict: a module can
hold several trunks and take several tensors. What changes is not the maths --
the weights are identical to the models already trained -- but the packaging:
one graph, one file, one compile unit, one AIPU resource allocation instead of
three.

    inputs : head_crop [N,3,128,128], hand_crops [M,3,128,128]
    outputs: head_cap [N], mask [N], gloves [M]

Both inputs carry a dynamic batch axis, because the number of people in a frame
is not known at compile time.
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchvision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import assert_contract

assert_contract()
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="runs/twotrunk/resnet34")
ap.add_argument("--out", default="runs/packed/apd_multinet.onnx")
ap.add_argument("--px", type=int, default=128)
ap.add_argument("--opset", type=int, default=17)
a = ap.parse_args()


def trunk(arch, nout):
    b = getattr(torchvision.models, arch)(weights=None)
    if hasattr(b, "classifier") and isinstance(b.classifier, nn.Sequential):
        d = b.classifier[-1].in_features; b.classifier = nn.Identity()
    else:
        d = b.fc.in_features; b.fc = nn.Identity()
    return nn.Sequential(), b, nn.Linear(d, nout)


class MultiNet(nn.Module):
    """Two independent trunks, one module, two inputs, three outputs.

    Independent on purpose: head and hand crops are different distributions, and
    forcing them through shared features measurably cost accuracy. Packaging
    them together changes deployment, not learning.
    """
    def __init__(self, arch_head, arch_hand):
        super().__init__()
        _, self.head_trunk, self.head_fc = trunk(arch_head, 2)   # head_cap, mask
        _, self.hand_trunk, self.hand_fc = trunk(arch_hand, 1)   # gloves

    def forward(self, head_crops, hand_crops):
        h = self.head_fc(self.head_trunk(head_crops))
        g = self.hand_fc(self.hand_trunk(hand_crops))
        return torch.sigmoid(h[:, 0]), torch.sigmoid(h[:, 1]), torch.sigmoid(g[:, 0])


hd = torch.load(Path(a.src) / "head_best.pt", map_location="cpu", weights_only=False)
hn = torch.load(Path(a.src) / "hand_best.pt", map_location="cpu", weights_only=False)
net = MultiNet(hd["arch"], hn["arch"])

def load_into(trunk_mod, fc_mod, sd):
    tk = {k[len("trunk."):]: v for k, v in sd.items() if k.startswith("trunk.")}
    fk = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
    trunk_mod.load_state_dict(tk); fc_mod.load_state_dict(fk)

load_into(net.head_trunk, net.head_fc, hd["model"])
load_into(net.hand_trunk, net.hand_fc, hn["model"])
net.eval()
print(f"packed {hd['arch']} (head: {hd['attrs']}) + {hn['arch']} (hand: {hn['attrs']})")
print(f"  params: {sum(p.numel() for p in net.parameters())/1e6:.1f}M")

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
dummy_h = torch.randn(3, 3, a.px, a.px)
dummy_d = torch.randn(6, 3, a.px, a.px)
torch.onnx.export(
    net, (dummy_h, dummy_d), str(out),
    input_names=["head_crops", "hand_crops"],
    output_names=["head_cap", "mask", "gloves"],
    dynamic_axes={"head_crops": {0: "n_people"}, "hand_crops": {0: "n_hands"},
                  "head_cap": {0: "n_people"}, "mask": {0: "n_people"},
                  "gloves": {0: "n_hands"}},
    opset_version=a.opset,
)
# fold external weights back in: a compiler that is handed one file should not
# also need a sidecar .onnx.data next to it
import onnx
_m = onnx.load(str(out))
onnx.save_model(_m, str(out), save_as_external_data=False)
Path(str(out) + ".data").unlink(missing_ok=True)
print(f"  exported -> {out}  ({out.stat().st_size/1e6:.1f} MB)")

# --- verify the packed graph equals the separate models it came from
import onnxruntime as ort
sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
print("\n  ONNX graph signature:")
for i in sess.get_inputs():  print(f"    in  {i.name:12} {i.shape}")
for o in sess.get_outputs(): print(f"    out {o.name:12} {o.shape}")

xh = np.random.randn(5, 3, a.px, a.px).astype(np.float32)
xd = np.random.randn(9, 3, a.px, a.px).astype(np.float32)
got = sess.run(None, {"head_crops": xh, "hand_crops": xd})
with torch.no_grad():
    ref = net(torch.from_numpy(xh), torch.from_numpy(xd))
diff = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref))
print(f"\n  max |onnx - torch| = {diff:.2e}   ({'identical' if diff < 1e-4 else 'MISMATCH'})")
print(f"  variable batch works: {got[0].shape[0]} people, {got[2].shape[0]} hands in one call")
