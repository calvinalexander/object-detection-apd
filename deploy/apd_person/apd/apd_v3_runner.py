"""Run the compiled apd_v3 off-device, via a shim for Axelera's custom ops.

The exported TorchScript calls into `qtools_cpp_ext`, a proprietary extension not
available outside the Axelera SDK. Those ops are standard fake-quantisation
primitives, so they are reimplemented here and the graph loads and runs on CPU or
GPU. Two are approximations: `lut_based_op` uses the true activation instead of
Axelera's lookup table, and `round_float18` keeps full precision instead of
rounding to their intermediate format. Both approximate the function the table
was built from, so this sits between the FP32 original and the INT8 device build.

The head is raw YOLOv8/11: three scales, DFL box regression in 64 channels
(4 sides x 16 bins) plus 7 class logits, decoded here rather than by the device's
libdecode_yolov8.so.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch

LABELS = ["person","gloves","head_cap","mask","no_gloves","no_head_cap","no_mask"]
STRIDES = (8, 16, 32)
REG_MAX = 16

def _install_shim():
    shim = Path(__file__).resolve().parents[1] / ".work"
    if str(shim) not in sys.path: sys.path.insert(0, str(shim))
    import qtools_shim
    qtools_shim.install()

def load(path: str, device: str = "cuda"):
    _install_shim()
    m = torch.jit.load(path, map_location=device).eval()
    return m.to(device)

@torch.no_grad()
def infer(model, chw: np.ndarray, device: str = "cuda") -> np.ndarray:
    """chw: (1,3,640,640) float32 in 0..1. Returns (N, 4+7) as cx,cy,w,h,scores."""
    outs = model(torch.from_numpy(chw).to(device))
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    box = [o for o in outs if o.shape[1] == 64]
    cls = [o for o in outs if o.shape[1] == len(LABELS)]
    box.sort(key=lambda t: -t.shape[-1]); cls.sort(key=lambda t: -t.shape[-1])

    rows = []
    for b, c, s in zip(box, cls, STRIDES):
        _, _, h, w = b.shape
        # DFL: 16 bins per side, softmaxed and expectation-weighted
        d = b.view(1, 4, REG_MAX, h * w).softmax(2)
        proj = torch.arange(REG_MAX, device=d.device, dtype=d.dtype).view(1, 1, REG_MAX, 1)
        dist = (d * proj).sum(2)                                   # (1,4,hw) in cells
        gy, gx = torch.meshgrid(torch.arange(h, device=d.device),
                                torch.arange(w, device=d.device), indexing="ij")
        ax = (gx.reshape(-1) + 0.5) * s; ay = (gy.reshape(-1) + 0.5) * s
        x1 = ax - dist[0, 0] * s; y1 = ay - dist[0, 1] * s
        x2 = ax + dist[0, 2] * s; y2 = ay + dist[0, 3] * s
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1), (y2 - y1)
        sc = c.view(1, len(LABELS), h * w)[0].sigmoid().T            # (hw,7)
        rows.append(torch.cat([torch.stack([cx, cy, bw, bh], 1), sc], 1))
    return torch.cat(rows, 0).cpu().numpy()
