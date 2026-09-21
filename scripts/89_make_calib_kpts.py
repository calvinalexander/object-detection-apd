"""Calibration set for compiling our pose network on the AIPU.

KptDataAdapter accepts only COCO keypoint JSON, and the built-in COCO2017 path
tries to DOWNLOAD the real dataset -- not something an edge device should do.
So the calibration set is generated here: real kitchen frames from apd_v3 val,
with COCO-17 keypoints produced by yolo11x-pose. Quantisation only needs
representative pixels; the annotations exist to satisfy the adapter's loader.
"""
import argparse, json, sys, shutil
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="artifacts/sdk/sdk/customers/apd_v3/dataset/images/val")
ap.add_argument("--out", default="deploy/apd_ours/dataset")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--conf", type=float, default=0.30)
a = ap.parse_args()

out = Path(a.out); img_dir = out / "images"
if out.exists(): shutil.rmtree(out)
img_dir.mkdir(parents=True)
srcs = sorted(p for p in Path(a.src).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
step = max(1, len(srcs) // a.n)
srcs = srcs[::step][:a.n]
pose = YOLO(".work/yolo11x-pose.pt")

KPT_NAMES = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
             "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
             "left_knee","right_knee","left_ankle","right_ankle"]
SKELETON = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],
            [2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
coco = {"images": [], "annotations": [],
        "categories": [{"id": 1, "name": "person", "supercategory": "person",
                        "keypoints": KPT_NAMES, "skeleton": SKELETON}]}
aid = 1
for iid, p in enumerate(srcs, 1):
    im = cv2.imread(str(p)); H, W = im.shape[:2]
    name = f"{iid:04d}_{p.stem}.jpg"
    cv2.imwrite(str(img_dir / name), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    coco["images"].append({"id": iid, "file_name": f"images/{name}", "width": W, "height": H})
    r = pose.predict(im, imgsz=1280, conf=a.conf, verbose=False, device=0)[0]
    if r.keypoints is None: continue
    for box, kp in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()):
        x1, y1, x2, y2 = [float(v) for v in box]
        flat, nvis = [], 0
        for x, y, c in kp:
            v = 2 if c >= 0.35 else 0
            nvis += v > 0
            flat += [float(x), float(y), v]
        coco["annotations"].append({"id": aid, "image_id": iid, "category_id": 1, "iscrowd": 0,
                                    "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1),
                                    "keypoints": flat, "num_keypoints": int(nvis)})
        aid += 1
(out / "calib.json").write_text(json.dumps(coco))
print(f"{len(coco['images'])} images, {len(coco['annotations'])} person annotations -> {out}/calib.json")
print(f"size: {sum(f.stat().st_size for f in img_dir.iterdir())/1e6:.1f} MB")
