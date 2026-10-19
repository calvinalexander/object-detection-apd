import cv2
import numpy as np
import threading
import queue
import os
from collections import defaultdict, deque
from datetime import datetime
from tlabcode.tracker import Kalman1D
from tlabcode.visualizer import draw_violation_highlight
from tlabcode.config import LOG, VIOLATION_IMAGES_DIR

# Global Buffer for Violation Frames
VIOLATION_RESIZE_WIDTH = 1024
MAX_VIOLATION_FRAMES = 3
FRAME_BUFFER = defaultdict(lambda: deque(maxlen=MAX_VIOLATION_FRAMES))


class SmartLineCrossingCounter:
    def __init__(
        self,
        line_points,
        db_manager=None,
        video_session_id=None,
        cfg=None,
    ):
        self.line_points = line_points
        self.video_session_id = video_session_id
        self.db = db_manager
        self.cfg = cfg

        if not line_points or len(line_points) != 2:
            LOG.error(f"Invalid line_points: {line_points}")
            raise ValueError("line_points must be [(x1,y1), (x2,y2)]")

        x1, y1 = line_points[0]
        x2, y2 = line_points[1]

        line_length = np.hypot(x2 - x1, y2 - y1)
        if line_length < 1.0:
            LOG.error(f"Line too short: {line_points} -> length={line_length}")
            raise ValueError(f"Counting line must be at least 1 pixel long")

        LOG.info(
            f"Counting line initialized: {line_points} -> length={line_length:.1f}px"
        )

        self.counted_objects = {}
        self.last_direction = {}  # arah terakhir yang dihitung per tray
        self.anchor_side = {}  # sisi garis terakhir tempat tray jelas berada
        self.class_in_counts = defaultdict(int)
        self.class_out_counts = defaultdict(int)
        self.violation_counts = defaultdict(int)

        self.line_vec = self._line_vector()
        self.normal = (-self.line_vec[1], self.line_vec[0])
        x1, y1 = self.line_points[0]
        x2, y2 = self.line_points[1]
        self.length = np.hypot(x2 - x1, y2 - y1) or 1.0

        self.kalman_dist = {}
        self.side_history = {}

        self.violation_clip_queue = queue.Queue()
        self.clip_writer_thread = threading.Thread(
            target=self._violation_clip_worker, daemon=True
        )
        self.clip_writer_thread.start()

    def update_line_geometry(self, new_line_points):
        """Called by main loop for Hot Reloading"""
        if self.line_points == new_line_points:
            return False
        LOG.info(f"HOT RELOAD: Updating counting line to {new_line_points}")
        self.line_points = new_line_points
        self.line_vec = self._line_vector()
        self.normal = (-self.line_vec[1], self.line_vec[0])
        x1, y1 = self.line_points[0]
        x2, y2 = self.line_points[1]
        self.length = np.hypot(x2 - x1, y2 - y1) or 1.0
        return True

    def _line_vector(self):
        dx = self.line_points[1][0] - self.line_points[0][0]
        dy = self.line_points[1][1] - self.line_points[0][1]
        L = np.hypot(dx, dy) or 1.0
        return (dx / L, dy / L)

    # def signed_distance(self, pt):
    #     x1, y1 = self.line_points[0]
    #     x2, y2 = self.line_points[1]
    #     px, py = pt
    #     return ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / self.length
    def signed_distance(self, pt):
        """Calculate signed distance from point to line with safety checks."""
        if self.length < 0.01:
            LOG.warning(f"Line length too small: {self.length}")
            return 0.0

        x1, y1 = self.line_points[0]
        x2, y2 = self.line_points[1]
        px, py = pt

        distance = ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / self.length

        # Sanity check: distance shouldn't exceed image dimensions
        if abs(distance) > 10000:
            LOG.warning(
                f"Abnormal distance {distance:.1f} for point {pt}, line {self.line_points}"
            )

        return distance

    def _init_obj(self, obj_id, first_dist):
        self.kalman_dist[obj_id] = Kalman1D(first_dist)
        self.side_history[obj_id] = deque(maxlen=15)

    def update(
        self, tracked_objects, trajectories, cfg, frame_num, frame_img=None, fps=15
    ):
        frame_events = []

        # --- 1. CALCULATE GLOBAL FLOW DIRECTION ---
        # Menghitung rata-rata arah pergerakan seluruh objek di layar
        total_dx, total_dy = 0.0, 0.0
        count_flow = 0

        for oid, tr in trajectories.items():
            # Hanya ambil objek yang sudah punya jejak gerak cukup
            if len(tr) > 2:
                # Ambil delta gerakan terakhir
                dx = tr[-1][0] - tr[-2][0]
                dy = tr[-1][1] - tr[-2][1]
                total_dx += dx
                total_dy += dy
                count_flow += 1

        if count_flow > 0:
            avg_flow_dx = total_dx / count_flow
            avg_flow_dy = total_dy / count_flow
        else:
            # Jika tidak ada objek, gunakan default 0
            avg_flow_dx, avg_flow_dy = 0.0, 0.0

        # Debug counters
        debug_stats = {
            "total_tracked": len(tracked_objects),
            "flow_dir": f"({avg_flow_dx:.1f}, {avg_flow_dy:.1f})",
            "skipped_small_area": 0,
            "skipped_short_traj": 0,
            "skipped_already_counted": 0,
            "skipped_no_side_change": 0,
            "skipped_velocity": 0,
            "successfully_counted": 0,
        }

        # Buffer Frames
        if frame_img is not None:
            h, w = frame_img.shape[:2]
            new_w = VIOLATION_RESIZE_WIDTH
            small_frame = cv2.resize(frame_img, (new_w, int(h * new_w / w)))
            for obj_id, data in tracked_objects.items():
                if data["cls"] not in cfg.VALID_OUT_CLASSES:
                    FRAME_BUFFER[obj_id].append(small_frame.copy())

        # Cleanup Old Objects
        for oid in list(self.counted_objects):
            if (
                frame_num - self.counted_objects[oid]["frame"]
                > cfg.ANTI_BOUNCE_COOLDOWN
            ):
                self.counted_objects.pop(oid, None)

        # --- 2. MAIN LOGIC LOOP ---
        for obj_id, data in tracked_objects.items():
            bbox = data["bbox"]
            cls = data["cls"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # Check 1: Area threshold
            if area < cfg.MIN_TRAY_AREA:
                debug_stats["skipped_small_area"] += 1
                continue

            traj = trajectories.get(obj_id)

            # Check 2: Trajectory length
            if not traj or len(traj) < cfg.MIN_TRAJECTORY_POINTS:
                debug_stats["skipped_short_traj"] += 1
                # Optional logging for debug
                # traj_len = len(traj) if traj else 0
                # LOG.info(f"Obj {obj_id}: Traj length {traj_len} < {cfg.MIN_TRAJECTORY_POINTS}")
                continue

            # Check 3: Already counted
            if obj_id in self.counted_objects:
                debug_stats["skipped_already_counted"] += 1
                continue

            centre = traj[-1]
            raw_d = self.signed_distance(centre)

            if abs(raw_d) > 5000:
                LOG.warning(
                    f"Obj {obj_id}: Abnormal raw distance {raw_d:.1f} - skipping"
                )
                continue

            if obj_id not in self.kalman_dist:
                self._init_obj(obj_id, raw_d)
                continue

            smooth_d = self.kalman_dist[obj_id].update(raw_d)

            # --- PENYEBERANGAN GARIS: SISI TERAKHIR + MARGIN SEPEREMPAT TRAY ---
            # Tray dihitung menyeberang setelah jelas berada di sisi lain: seperempat panjang tray
            # melewati garis (minimal CROSSING_BUFFER_ZONE). Sisi asal diingat selama tray terlacak,
            # jadi tray yang berhenti, macet, atau maju mundur di atas garis dihitung sekali,
            # saat benar-benar meninggalkan garis.
            x1, y1, x2, y2 = bbox
            ext = abs((x2 - x1) * self.normal[0]) + abs((y2 - y1) * self.normal[1])
            margin = max(cfg.CROSSING_BUFFER_ZONE, 0.25 * ext)
            side = -1 if smooth_d < -margin else 1 if smooth_d > margin else 0

            if obj_id not in self.anchor_side:
                # Pertama terlihat: pakai sisi tempat tray berada, walau belum jauh dari garis
                # (tray yang sudah di atas garis saat video mulai tetap punya sisi asal).
                if abs(smooth_d) > cfg.CROSSING_BUFFER_ZONE:
                    self.anchor_side[obj_id] = 1 if smooth_d > 0 else -1
                continue

            if side == 0 or side == self.anchor_side[obj_id] or len(traj) < 2:
                debug_stats["skipped_no_side_change"] += 1
                continue

            # Validasi kecepatan: tray harus bergerak, dan tidak sejajar garis.
            # Jika gagal, dicoba lagi di frame berikutnya (sisi asal tetap diingat).
            obj_dx = traj[-1][0] - traj[-2][0]
            obj_dy = traj[-1][1] - traj[-2][1]
            vnorm = np.hypot(obj_dx, obj_dy)
            if vnorm <= 0.5:
                debug_stats["skipped_velocity"] += 1
                continue
            perp = abs(obj_dx * self.normal[0] + obj_dy * self.normal[1]) / vnorm
            if perp < 0.2:
                debug_stats["skipped_velocity"] += 1
                continue

            # --- PENENTUAN ARAH: SISI ASAL -> SISI TUJUAN ---
            # Bukan dari arus rata-rata global (arus global salah jika nampan yang dikembalikan
            # adalah satu-satunya objek yang bergerak). Sisi positif = sisi yang ditunjuk normal
            # garis = "OUT". Jika arah terbalik di suatu kamera, tukar urutan titik garis ROI.
            prev_side = self.anchor_side[obj_id]
            self.anchor_side[obj_id] = side
            direction = "out" if side > 0 else "in"
            determination_method = f"SIDE_CHANGE ({prev_side:+d} -> {side:+d}, margin {margin:.0f}px)"

            # Tray yang sama tidak bisa dihitung dua kali berturut-turut ke arah yang sama.
            if self.last_direction.get(obj_id) == direction:
                continue
            self.last_direction[obj_id] = direction

            # ✅ COUNT IT!
            if direction == "in":
                self.class_in_counts[cls] += 1
            else:
                self.class_out_counts[cls] += 1

            debug_stats["successfully_counted"] += 1
            frame_events.append({"direction": direction, "cls": cls})

            LOG.info(
                f"✅ COUNTED: Obj {obj_id} | {direction.upper()} | Method: {determination_method} | "
                f"Class: {cls} | Dist: {smooth_d:.1f} | "
                f"Total IN: {sum(self.class_in_counts.values())} | "
                f"Total OUT: {sum(self.class_out_counts.values())}"
            )

            # Handle Violation Logic (Hanya jika arah OUT dan Kelas salah)
            if (
                direction == "out"
                and cls not in cfg.VALID_OUT_CLASSES
                and frame_img is not None
            ):
                self._log_violation(
                    frame_num,
                    obj_id,
                    cls,
                    frame_img,
                    cfg,
                    direction,
                    tracked_objects,
                    fps,
                )
                self.violation_counts[cls] += 1

            # Add to counted list
            self.counted_objects[obj_id] = {
                "direction": direction,
                "frame": frame_num,
            }

        # Log debug stats every 30 frames
        if frame_num % 30 == 0 and debug_stats["total_tracked"] > 0:
            LOG.info(f"Frame {frame_num} Debug Stats: {debug_stats}")

        return frame_events

    def _log_violation(
        self,
        frame_num,
        obj_id,
        cls,
        frame_img,
        cfg,
        direction,
        tracked_objects,
        fps,
    ):
        highlighted = frame_img.copy()
        if obj_id in tracked_objects:
            draw_violation_highlight(
                highlighted,
                obj_id,
                tracked_objects,
                cfg.CLASS_COLORS,
                cfg,
            )

        h, w = highlighted.shape[:2]
        new_w = VIOLATION_RESIZE_WIDTH
        small_trigger = cv2.resize(highlighted, (new_w, int(h * new_w / w)))

        frames = list(FRAME_BUFFER[obj_id])
        frames.append(small_trigger)

        self.violation_clip_queue.put(
            (obj_id, frames, fps, frame_num, cls, self.video_session_id)
        )

    def _violation_clip_worker(self):
        while True:
            item = self.violation_clip_queue.get()
            if item is None:
                break
            obj_id, frames, fps, trigger_frame, cls, sess_id = item

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"violation_{self.cfg.EDGE_DEVICE_ID}_{timestamp}_id{obj_id}_f{trigger_frame}.jpg"
            img_path = os.path.join(self.cfg.VIOLATION_IMAGES_DIR, img_name)

            if frames:
                cv2.imwrite(img_path, frames[-1], [cv2.IMWRITE_JPEG_QUALITY, 95])

            if self.db:
                self.db.log_violation(
                    sess_id,
                    trigger_frame,
                    fps,
                    obj_id,
                    cls,
                    self.cfg.CLASS_NAMES.get(cls, "unknown"),
                    img_path,
                    self.cfg.SPPG_ID,
                    self.cfg.EDGE_DEVICE_ID,
                    self.cfg.CAMERA_ID,
                )
            self.violation_clip_queue.task_done()

    def get_counts(self):
        return dict(self.class_in_counts), dict(self.class_out_counts)

    def close(self):
        self.violation_clip_queue.put(None)
        self.clip_writer_thread.join(timeout=5)
