import numpy as np
from collections import deque


class Kalman1D:
    def __init__(self, init_val, process_noise=5e-3, measurement_noise=2e-1):
        self.x = init_val
        self.P = 1.0
        self.Q = process_noise
        self.R = measurement_noise

    def update(self, measurement):
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred
        return self.x


class AdvancedObjectTracker:
    def __init__(self, max_disappeared=30, track_history_len=30, debug_logger=None):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.track_history_len = track_history_len
        self.trajectories = {}
        self.velocities = {}
        self.counter = None  # Assigned in main

    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        xi1, yi1 = max(x1, x1_), max(y1, y1_)
        xi2, yi2 = min(x2, x2_), min(y2, y2_)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - inter
        return inter / union if union > 0 else 0

    def euclidean_distance(self, p1, p2):
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_bottom_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, bbox[3])

    def calculate_velocity(self, trajectory, window=5):
        if len(trajectory) < 2:
            return (0, 0)
        pts = list(trajectory)[-min(window, len(trajectory)) :]
        if len(pts) < 2:
            return (0, 0)
        vx = sum(pts[i][0] - pts[i - 1][0] for i in range(1, len(pts)))
        vy = sum(pts[i][1] - pts[i - 1][1] for i in range(1, len(pts)))
        n = len(pts) - 1
        return (vx / n, vy / n)

    def predict_position(self, obj_id, steps=1):
        if obj_id not in self.trajectories or not self.trajectories[obj_id]:
            return None
        last = list(self.trajectories[obj_id])[-1]
        vel = self.velocities.get(obj_id, (0, 0))
        return (last[0] + vel[0] * steps, last[1] + vel[1] * steps)

    def register(self, bbox, cls, frame_num):
        obj_id = self.next_id
        self.objects[obj_id] = {
            "bbox": bbox,
            "cls": cls,
            "birth_frame": frame_num,
            "votes": {cls: 1},
        }
        self.disappeared[obj_id] = 0
        self.trajectories[obj_id] = deque(maxlen=self.track_history_len)
        self.trajectories[obj_id].append(self.get_bottom_center(bbox))
        self.velocities[obj_id] = (0, 0)
        self.next_id += 1
        return obj_id

    def deregister(self, obj_id, frame_num):
        for d in (self.objects, self.disappeared, self.trajectories, self.velocities):
            d.pop(obj_id, None)
        if self.counter:
            self.counter.kalman_dist.pop(obj_id, None)
            self.counter.side_history.pop(obj_id, None)

    def update(self, detections, cfg, frame_num):
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid, frame_num)
            return self.objects

        input_boxes = [d[:4] for d in detections]
        input_classes = [int(d[5]) for d in detections]
        input_centers = [self.get_bottom_center(b) for b in input_boxes]

        if not self.objects:
            for bbox, cls in zip(input_boxes, input_classes):
                self.register(bbox, cls, frame_num)
        else:
            obj_ids = list(self.objects.keys())
            obj_boxes = [self.objects[i]["bbox"] for i in obj_ids]
            obj_centers = [
                (
                    list(self.trajectories[i])[-1]
                    if self.trajectories[i]
                    else self.get_bottom_center(obj_boxes[idx])
                )
                for idx, i in enumerate(obj_ids)
            ]

            n_obj, n_det = len(obj_boxes), len(input_boxes)
            iou_mat = np.zeros((n_obj, n_det), dtype=np.float32)
            dist_mat = np.zeros((n_obj, n_det), dtype=np.float32)
            pred_mat = np.zeros((n_obj, n_det), dtype=np.float32)

            for i, (obox, ocent, oid) in enumerate(
                zip(obj_boxes, obj_centers, obj_ids)
            ):
                pred = self.predict_position(oid, 1)
                # Bandingkan dengan kotak prediksi (kotak terakhir digeser sesuai kecepatan
                # x jumlah frame sejak terakhir terlihat), bukan kotak terakhir yang diam.
                gap = self.disappeared[oid] + 1
                vx, vy = self.velocities.get(oid, (0, 0))
                pbox = (
                    obox[0] + vx * gap,
                    obox[1] + vy * gap,
                    obox[2] + vx * gap,
                    obox[3] + vy * gap,
                )
                for j, (ibox, icent) in enumerate(zip(input_boxes, input_centers)):
                    iou_mat[i, j] = self.iou(pbox, ibox)
                    dist_mat[i, j] = self.euclidean_distance(ocent, icent)
                    pred_mat[i, j] = (
                        self.euclidean_distance(pred, icent) if pred else dist_mat[i, j]
                    )

            max_d = max(np.max(dist_mat), 1)
            max_pd = max(np.max(pred_mat), 1)
            norm_d = 1 - (dist_mat / max_d)
            norm_pd = 1 - (pred_mat / max_pd)
            score = 0.5 * iou_mat + 0.25 * norm_d + 0.25 * norm_pd

            # Cocokkan lintas kelas: kelas isi nampan yang berkedip antar-frame
            # bukan nampan baru. Kelas akhir ditentukan lewat voting mayoritas.
            # Track yang terlihat di frame sebelumnya dicocokkan lebih dulu, supaya
            # track basi (nampan yang sudah lewat) tidak merebut ID nampan berikutnya
            # saat conveyor cepat.
            matches = []
            for i in range(n_obj):
                live = self.disappeared[obj_ids[i]] == 0
                for j in range(n_det):
                    matches.append(
                        (live, score[i, j], i, j, iou_mat[i, j], dist_mat[i, j])
                    )
            matches.sort(reverse=True)

            used_det = set()
            used_obj = set()
            for _, sc, i, j, iou, dist in matches:
                oid = obj_ids[i]
                jump = dist_mat[i, j]
                if (
                    sc > 0.3
                    and iou > cfg.IOU_THRESHOLD
                    and dist < cfg.MAX_DISTANCE
                    and jump < cfg.MAX_POSITION_JUMP
                    and j not in used_det
                    and i not in used_obj
                ):
                    self.objects[oid]["bbox"] = input_boxes[j]
                    votes = self.objects[oid]["votes"]
                    votes[input_classes[j]] = votes.get(input_classes[j], 0) + 1
                    self.objects[oid]["cls"] = max(votes, key=votes.get)
                    self.disappeared[oid] = 0
                    new_c = input_centers[j]
                    self.trajectories[oid].append(new_c)
                    self.velocities[oid] = self.calculate_velocity(
                        self.trajectories[oid], cfg.VELOCITY_SMOOTHING_WINDOW
                    )
                    used_det.add(j)
                    used_obj.add(i)

            for i, oid in enumerate(obj_ids):
                if i not in used_obj:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid, frame_num)

            for j in range(n_det):
                if j not in used_det:
                    self.register(input_boxes[j], input_classes[j], frame_num)

        return self.objects
