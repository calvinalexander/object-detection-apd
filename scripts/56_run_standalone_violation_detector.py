import os
import sys
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

# Add local path
sys.path.append("/home/ubuntu/tryan/apd")
from apd.standalone_detector import StandaloneViolationDetector
from apd.pose import is_frontal, head_roi, hand_rois


# EfficientNet Crop Classifier (Self-Contained)
effnet_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CropClassifier:
    def __init__(self, attr_name: str, model_type: str = "effnet_b1"):
        self.attr_name = attr_name
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = Path(f"runs/classify_effnet_b1/{attr_name}_best.pt")
        self.model = models.efficientnet_b1()
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)
        if ckpt_path.exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.dev))
        self.model = self.model.to(self.dev).eval()

    def predict_crop(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.5, 0.0
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = effnet_transform(pil_img).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
        return prob, 1.0


def run_standalone_test(
    video_path: str,
    output_dir: str = "artifacts/standalone_test",
    engine: str = "yolo_7cls",  # "yolo_7cls" or "edge_ensemble"
    yolo_model_path: str = "/home/ubuntu/tryan/apd/runs/_retired/voter3_yolov8s/weights/best.pt",
    max_seconds: float = 15.0,
):
    print(f"\n========================================================")
    print(f"🎬 STANDALONE VIOLATION CAPTURE TEST")
    print(f"Input Video: {video_path}")
    print(f"Engine Mode: {engine}")
    print(f"Output Directory: {output_dir}")
    print(f"========================================================\n")

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps)

    out_video_path = os.path.join(output_dir, f"output_{engine}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    detector = StandaloneViolationDetector(
        output_dir=output_dir,
        device_id="ED-STANDALONE-01",
        camera_id="CAM01",
        vote_window=5,
        min_age_frames=3,
        min_violation_ratio=0.60,
        cooldown_ms=20000,
    )

    if engine == "yolo_7cls":
        yolo = YOLO(yolo_model_path)
        class_names = yolo.names
        print(f"Loaded 7-class YOLO model: {yolo_model_path}")
        print(f"Classes: {class_names}\n")

        frame_count = 0
        total_violations_captured = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = yolo(frame, verbose=False)[0]
            person_boxes = []
            person_confs = []
            apd_dets = []

            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cid = int(box.cls[0].cpu().numpy())
                cname = class_names.get(cid, str(cid))

                if cname == "person":
                    person_boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
                    person_confs.append(conf)
                else:
                    apd_dets.append({
                        "label": cname,
                        "bbox": (xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        "conf": conf
                    })

            rendered_frame, events = detector.process_frame_detections(
                frame, person_boxes, apd_dets, person_confs
            )
            total_violations_captured += len(events)
            writer.write(rendered_frame)

    elif engine == "edge_ensemble":
        pose_path = ".work/yolo11s-pose.pt" if os.path.exists(".work/yolo11s-pose.pt") else "yolo11s-pose.pt"
        pose_model = YOLO(pose_path)
        cls_head = CropClassifier("head_cap", "effnet_b1")
        cls_mask = CropClassifier("mask", "effnet_b1")
        cls_glove = CropClassifier("gloves", "effnet_b1")
        print("Loaded Two-Stage Edge Ensemble (YOLO11s Pose + EfficientNet Classifiers)")

        frame_count = 0
        total_violations_captured = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = pose_model(frame, verbose=False)[0]
            person_boxes = []
            person_confs = []
            apd_dets = []

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                kpts = results.keypoints.data.cpu().numpy() if results.keypoints is not None else None

                for idx, (b, c) in enumerate(zip(boxes, confs)):
                    if c < 0.35:
                        continue
                    px1, py1, px2, py2 = map(int, b[:4])
                    person_boxes.append((px1, py1, px2, py2))
                    person_confs.append(float(c))

                    if kpts is not None and idx < len(kpts):
                        kp = kpts[idx]
                        frontal = is_frontal(kp)
                        h_roi = head_roi(kp, (px1, py1, px2, py2))
                        if h_roi is not None:
                            hx1, hy1, hx2, hy2 = map(int, h_roi)
                            hx1, hy1 = max(0, hx1), max(0, hy1)
                            hx2, hy2 = min(width, hx2), min(height, hy2)
                            if hx2 > hx1 and hy2 > hy1:
                                head_crop = frame[hy1:hy2, hx1:hx2]
                                p_cap, _ = cls_head.predict_crop(head_crop)
                                if p_cap < 0.10:
                                    apd_dets.append({"label": "no_head_cap", "bbox": (hx1, hy1, hx2, hy2), "conf": 1.0 - p_cap})
                                elif p_cap > 0.90:
                                    apd_dets.append({"label": "head_cap", "bbox": (hx1, hy1, hx2, hy2), "conf": p_cap})

                                if frontal:
                                    p_mask, _ = cls_mask.predict_crop(head_crop)
                                    if p_mask < 0.10:
                                        apd_dets.append({"label": "no_mask", "bbox": (hx1, hy1, hx2, hy2), "conf": 1.0 - p_mask})
                                    elif p_mask > 0.90:
                                        apd_dets.append({"label": "mask", "bbox": (hx1, hy1, hx2, hy2), "conf": p_mask})

                        h_rois = hand_rois(kp, (px1, py1, px2, py2))
                        for hand in h_rois:
                            gx1, gy1, gx2, gy2 = map(int, hand)
                            gx1, gy1 = max(0, gx1), max(0, gy1)
                            gx2, gy2 = min(width, gx2), min(height, gy2)
                            if gx2 > gx1 and gy2 > gy1:
                                glove_crop = frame[gy1:gy2, gx1:gx2]
                                p_glove, _ = cls_glove.predict_crop(glove_crop)
                                if p_glove < 0.10:
                                    apd_dets.append({"label": "no_gloves", "bbox": (gx1, gy1, gx2, gy2), "conf": 1.0 - p_glove})
                                elif p_glove > 0.90:
                                    apd_dets.append({"label": "gloves", "bbox": (gx1, gy1, gx2, gy2), "conf": p_glove})

            rendered_frame, events = detector.process_frame_detections(
                frame, person_boxes, apd_dets, person_confs
            )
            total_violations_captured += len(events)
            writer.write(rendered_frame)

    cap.release()
    writer.release()

    print(f"\n========================================================")
    print(f"✅ TEST COMPLETED")
    print(f"Processed Frames: {frame_count}")
    print(f"Total Violations Captured & Saved Locally: {total_violations_captured}")
    print(f"Output Video: {out_video_path}")
    print(f"SQLite DB: {os.path.join(output_dir, 'violations.db')}")
    print(f"JSON Log: {os.path.join(output_dir, 'violations.json')}")
    print(f"Snapshots Directory: {os.path.join(output_dir, 'snapshots', 'CAM01')}")
    print(f"========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/recording/Recording 2026-09-11 113142.mp4")
    parser.add_argument("--outdir", type=str, default="artifacts/standalone_violations")
    parser.add_argument("--engine", type=str, default="yolo_7cls", choices=["yolo_7cls", "edge_ensemble"])
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args()

    run_standalone_test(
        video_path=args.video,
        output_dir=args.outdir,
        engine=args.engine,
        max_seconds=args.seconds
    )
