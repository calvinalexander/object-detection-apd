"""What each bake-off winner costs to run, per person and per frame.

Accuracy alone does not pick the model: the classifier runs once per head and
twice per pair of hands, so a frame with ten workers pays for roughly thirty
crops. Batched, since that is how a deployment would issue them.
"""
import statistics, sys, time
from pathlib import Path
import numpy as np, torch, torchvision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO

dev = "cuda"
def torch_model(arch, path):
    net = getattr(torchvision.models, arch)(weights=None)
    if hasattr(net, "classifier") and isinstance(net.classifier, torch.nn.Sequential):
        net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
    elif hasattr(net, "fc"):
        net.fc = torch.nn.Linear(net.fc.in_features, 1)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return net.to(dev).eval()

CAND = {
    "effnet-b1 (published)": ("torch","efficientnet_b1","runs/effnet_coco_holdout/head_cap_last.pt"),
    "effnet-b0 (regularised)": ("torch","efficientnet_b0","runs/bakeoff_torch/efficientnet_b0/head_cap_best.pt"),
    "resnet34 (regularised)": ("torch","resnet34","runs/bakeoff_torch/resnet34/head_cap_best.pt"),
    "yolo26n-cls": ("yolo",None,"runs/classify/runs/bakeoff/yolo26n-cls_head_cap/weights/best.pt"),
    "yolo11s-cls": ("yolo",None,"runs/classify/runs/bakeoff/yolo11s-cls_head_cap/weights/best.pt"),
}
BATCH = 30          # ~10 workers: 10 heads + 20 hands
x = torch.randn(BATCH, 3, 128, 128, device=dev)
crops = [np.random.randint(0, 255, (128,128,3), dtype=np.uint8) for _ in range(BATCH)]

print(f"{'model':26} {'params':>9} {'fp32 MB':>8} {'batch-30 ms':>12} {'ms/crop':>8} {'crops/s':>9}")
for name, (kind, arch, path) in CAND.items():
    if not Path(path).exists():
        print(f"{name:26}  (missing)"); continue
    if kind == "torch":
        m = torch_model(arch, path)
        n = sum(p.numel() for p in m.parameters())
        for _ in range(5):
            with torch.no_grad(): m(x)
        torch.cuda.synchronize(); ts = []
        for _ in range(20):
            t = time.perf_counter()
            with torch.no_grad(): m(x)
            torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1000)
    else:
        m = YOLO(path)
        n = sum(p.numel() for p in m.model.parameters())
        for _ in range(3): m.predict(crops, imgsz=128, verbose=False, device=0)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            m.predict(crops, imgsz=128, verbose=False, device=0)
            ts.append((time.perf_counter()-t)*1000)
    ms = statistics.median(ts)
    print(f"{name:26} {n/1e6:8.2f}M {n*4/1e6:8.1f} {ms:12.2f} {ms/BATCH:8.3f} {1000/(ms/BATCH):9.0f}")

print(f"\nThree attributes need two forward passes per person (one head crop, one")
print(f"hand pair), so a 10-person frame is ~{BATCH} crops. Multiply ms/crop accordingly.")
