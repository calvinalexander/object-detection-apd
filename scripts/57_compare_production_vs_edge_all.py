import os
import sys
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import openvino as ov
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

# Add local path
sys.path.append("/home/ubuntu/tryan/apd")
from apd.standalone_detector import StandaloneViolationDetector, draw_table_overlay
from apd.pose import is_frontal, head_roi, hand_rois

# EfficientNet Crop Classifier (Self-Contained)
effnet_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DEV = "cuda" if torch.cuda.is_available() else "cpu"

class CropClassifier:
    def __init__(self, attr_name: str, model_type: str = "effnet_b1"):
        self.attr_name = attr_name
        self.dev = DEV
        ckpt_path = Path(f"runs/classify_effnet_b1/{attr_name}_best.pt")
        self.model = models.efficientnet_b1()
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)
        if ckpt_path.exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.dev))
        self.model = self.model.to(self.dev).eval()

    def predict_crop(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.5
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = effnet_transform(pil_img).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
        return prob


# OpenVINO Production Model Wrapper
class OpenVinoProductionModel:
    def __init__(self, model_xml="model/sppg/apd_mbg_openvino_model/apd_mbg_yolo11s.xml", conf_thresh=0.30):
        core = ov.Core()
        ovdev = "GPU" if "GPU" in core.available_devices else "CPU"
        self.model = core.compile_model(core.read_model(model_xml), ovdev)
        self.out_layer = self.model.outputs[0]
        self.conf_thresh = conf_thresh
        self.class_names = {0: 'person', 1: 'gloves', 2: 'head_cap', 3: 'mask', 4: 'no_gloves', 5: 'no_head_cap', 6: 'no_mask'}

    def detect(self, frame: np.ndarray):
        h_orig, w_orig = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, :]

        res = self.model(img)[self.out_layer]
        # Parse YOLO output shape [1, 11, 8400]
        output = res[0]
        if output.shape[0] < output.shape[1]:
            output = output.T

        person_boxes, person_confs, apd_dets = [], [], []

        for row in output:
            box = row[:4]
            scores = row[4:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < self.conf_thresh:
                continue

            cx, cy, w, h = box
            x1 = int((cx - w / 2) * w_orig / 640.0)
            y1 = int((cy - h / 2) * h_orig / 640.0)
            x2 = int((cx + w / 2) * w_orig / 640.0)
            y2 = int((cy + h / 2) * h_orig / 640.0)

            cname = self.class_names.get(cid, str(cid))
            if cname == "person":
                person_boxes.append((x1, y1, x2, y2))
                person_confs.append(conf)
            else:
                apd_dets.append({"label": cname, "bbox": (x1, y1, x2, y2), "conf": conf})

        return person_boxes, person_confs, apd_dets


def process_video_pair(
    video_path: str,
    output_path: str,
    prod_model: OpenVinoProductionModel,
    pose_model: YOLO,
    cls_head: CropClassifier,
    cls_mask: CropClassifier,
    cls_glove: CropClassifier,
    seconds: float = 10.0,
):
    print(f"\n========================================================")
    print(f"📹 PROCESSING RECORDING COMPARISON: {video_path}")
    print(f"Output: {output_path}")
    print(f"========================================================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(seconds * fps) if seconds > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = w_orig * 2
    out_h = h_orig
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    # Initialize two detector instances for local violation tracking
    detector_prod = StandaloneViolationDetector(output_dir="artifacts/compare_temp_prod", camera_id="PROD")
    detector_edge = StandaloneViolationDetector(output_dir="artifacts/compare_temp_edge", camera_id="EDGE")

    stats = {
        "prod": {"head_cap": {"ok": 0, "violation": 0, "unknown": 0}, "mask": {"ok": 0, "violation": 0, "unknown": 0}, "gloves": {"ok": 0, "violation": 0, "unknown": 0}, "persons": 0, "violations_captured": 0},
        "edge": {"head_cap": {"ok": 0, "violation": 0, "unknown": 0}, "mask": {"ok": 0, "violation": 0, "unknown": 0}, "gloves": {"ok": 0, "violation": 0, "unknown": 0}, "persons": 0, "violations_captured": 0},
    }

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ----------------------------------------------------
        # 1. Production Model (OpenVINO single-pass detector)
        # ----------------------------------------------------
        p_boxes, p_confs, p_apd = prod_model.detect(frame)
        frame_prod, events_p = detector_prod.process_frame_detections(frame.copy(), p_boxes, p_apd, p_confs)
        stats["prod"]["persons"] += len(p_boxes)
        stats["prod"]["violations_captured"] += len(events_p)

        # ----------------------------------------------------
        # 2. Edge Model (Pose + EfficientNet Classifiers 0.10/0.90)
        # ----------------------------------------------------
        e_boxes, e_confs, e_apd = [], [], []
        pose_res = pose_model(frame, verbose=False)[0]

        if pose_res.boxes is not None and len(pose_res.boxes) > 0:
            boxes = pose_res.boxes.xyxy.cpu().numpy()
            confs = pose_res.boxes.conf.cpu().numpy()
            kpts = pose_res.keypoints.data.cpu().numpy() if pose_res.keypoints is not None else None

            for idx, (b, c) in enumerate(zip(boxes, confs)):
                if c < 0.35:
                    continue
                px1, py1, px2, py2 = map(int, b[:4])
                e_boxes.append((px1, py1, px2, py2))
                e_confs.append(float(c))

                if kpts is not None and idx < len(kpts):
                    kp = kpts[idx]
                    frontal = is_frontal(kp)
                    h_roi = head_roi(kp, (px1, py1, px2, py2))
                    if h_roi is not None:
                        hx1, hy1, hx2, hy2 = map(int, h_roi)
                        hx1, hy1 = max(0, hx1), max(0, hy1)
                        hx2, hy2 = min(w_orig, hx2), min(h_orig, hy2)
                        if hx2 > hx1 and hy2 > hy1:
                            head_crop = frame[hy1:hy2, hx1:hx2]
                            p_cap = cls_head.predict_crop(head_crop)
                            if p_cap < 0.10:
                                e_apd.append({"label": "no_head_cap", "bbox": (hx1, hy1, hx2, hy2), "conf": 1.0 - p_cap})
                            elif p_cap > 0.90:
                                e_apd.append({"label": "head_cap", "bbox": (hx1, hy1, hx2, hy2), "conf": p_cap})

                            if frontal:
                                p_mask = cls_mask.predict_crop(head_crop)
                                if p_mask < 0.10:
                                    e_apd.append({"label": "no_mask", "bbox": (hx1, hy1, hx2, hy2), "conf": 1.0 - p_mask})
                                elif p_mask > 0.90:
                                    e_apd.append({"label": "mask", "bbox": (hx1, hy1, hx2, hy2), "conf": p_mask})

                    h_rois = hand_rois(kp, (px1, py1, px2, py2))
                    for hand in h_rois:
                        gx1, gy1, gx2, gy2 = map(int, hand)
                        gx1, gy1 = max(0, gx1), max(0, gy1)
                        gx2, gy2 = min(w_orig, gx2), min(h_orig, gy2)
                        if gx2 > gx1 and gy2 > gy1:
                            glove_crop = frame[gy1:gy2, gx1:gx2]
                            p_glove = cls_glove.predict_crop(glove_crop)
                            if p_glove < 0.10:
                                e_apd.append({"label": "no_gloves", "bbox": (gx1, gy1, gx2, gy2), "conf": 1.0 - p_glove})
                            elif p_glove > 0.90:
                                e_apd.append({"label": "gloves", "bbox": (gx1, gy1, gx2, gy2), "conf": p_glove})

        frame_edge, events_e = detector_edge.process_frame_detections(frame.copy(), e_boxes, e_apd, e_confs)
        stats["edge"]["persons"] += len(e_boxes)
        stats["edge"]["violations_captured"] += len(events_e)

        # Annotate Panels
        cv2.putText(frame_prod, "PRODUCTION MODEL (OpenVINO)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame_edge, "OPTIMIZED EDGE MODEL (Pose + EffNet 0.10/0.90)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Side-by-side stitch
        combined = np.hstack([frame_prod, frame_edge])
        writer.write(combined)

    cap.release()
    writer.release()

    mean_p_prod = round(stats["prod"]["persons"] / float(max(1, frame_idx)), 2)
    mean_p_edge = round(stats["edge"]["persons"] / float(max(1, frame_idx)), 2)

    print(f"\n📊 COMPARISON SUMMARY [{Path(video_path).name}]")
    print(f"Processed Frames: {frame_idx}")
    print(f"Mean Persons/Frame  --> Production: {mean_p_prod} | Edge: {mean_p_edge}")
    print(f"Violations Captured --> Production: {stats['prod']['violations_captured']} | Edge: {stats['edge']['violations_captured']}")
    print(f"Rendered Output Video: {output_path}\n")

    return {
        "video": Path(video_path).name,
        "frames": frame_idx,
        "prod_mean_persons": mean_p_prod,
        "edge_mean_persons": mean_p_edge,
        "prod_violations": stats["prod"]["violations_captured"],
        "edge_violations": stats["edge"]["violations_captured"],
        "output_path": output_path
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    recordings = [
        # Batch 1
        ("data/recording/Recording 2026-09-11 113142.mp4", "artifacts/compare_prod_vs_edge_rec113142.mp4"),
        ("data/recording/Recording 2026-09-11 113959.mp4", "artifacts/compare_prod_vs_edge_rec113959.mp4"),
        ("data/recording/Screen Recording 2026-09-11 113644.mp4", "artifacts/compare_prod_vs_edge_rec113644.mp4"),
        ("/data/adin/sppg-ai/cp/Recording 2026-09-09 104135.mp4", "artifacts/compare_prod_vs_edge_rec104135.mp4"),
        # Batch 2
        ("data/recording/batch_2/Recording 2026-09-11 133700.mp4", "artifacts/compare_prod_vs_edge_b2_rec133700.mp4"),
        ("data/recording/batch_2/Recording 2026-09-11 133958.mp4", "artifacts/compare_prod_vs_edge_b2_rec133958.mp4"),
        ("data/recording/batch_2/Recording 2026-09-11 134226.mp4", "artifacts/compare_prod_vs_edge_b2_rec134226.mp4"),
        # Batch 3
        ("data/recording/batch_3/Recording 2026-09-11 135526.mp4", "artifacts/compare_prod_vs_edge_b3_rec135526.mp4"),
        ("data/recording/batch_3/Recording 2026-09-11 135711.mp4", "artifacts/compare_prod_vs_edge_b3_rec135711.mp4"),
        ("data/recording/batch_3/Recording 2026-09-11 140115.mp4", "artifacts/compare_prod_vs_edge_b3_rec140115.mp4"),
    ]

    print(f"Loading Models...")
    prod_model = OpenVinoProductionModel()
    pose_path = ".work/yolo11s-pose.pt" if os.path.exists(".work/yolo11s-pose.pt") else "yolo11s-pose.pt"
    pose_model = YOLO(pose_path)
    cls_head = CropClassifier("head_cap", "effnet_b1")
    cls_mask = CropClassifier("mask", "effnet_b1")
    cls_glove = CropClassifier("gloves", "effnet_b1")

    all_results = []
    for vid_path, out_path in recordings:
        if os.path.exists(vid_path):
            res = process_video_pair(
                vid_path, out_path, prod_model, pose_model, cls_head, cls_mask, cls_glove, seconds=args.seconds
            )
            if res:
                all_results.append(res)

    print("\n========================================================")
    print("🏆 FINAL ALL-RECORDINGS COMPARISON SUMMARY TABLE")
    print("========================================================")
    print(f"{'Video Recording':<40} | {'Prod Persons/Fr':<16} | {'Edge Persons/Fr':<16} | {'Prod Violations':<15} | {'Edge Violations':<15}")
    print("-" * 110)
    for r in all_results:
        print(f"{r['video']:<40} | {r['prod_mean_persons']:<16} | {r['edge_mean_persons']:<16} | {r['prod_violations']:<15} | {r['edge_violations']:<15}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
