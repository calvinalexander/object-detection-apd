import os
import sys
import time
import json
import uuid
import sqlite3
from collections import deque, defaultdict
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add path to load apd_tracker if needed
sys.path.append("/data/adin/sppg-ai/src")
try:
    from apd_tracker.byte_tracker import BYTETracker
except ImportError:
    BYTETracker = None


class TrackerArgs:
    def __init__(self, track_thresh=0.30, track_buffer=60, match_thresh=0.8, aspect_ratio_thresh=1.6, min_box_area=10, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
        self.mot20 = mot20


# ---------------------------------------------------------
# APD Association Utilities (Extracted & Self-Contained)
# ---------------------------------------------------------
def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter + 1e-9)


def center_inside(apd_box: Tuple[int, int, int, int], person_box: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = apd_box
    px1, py1, px2, py2 = person_box
    cx = (ax1 + ax2) / 2.0
    cy = (ay1 + ay2) / 2.0
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)


def assign_apd_smart(
    active_tracks: List[dict],
    apd_dets: List[dict],
    class_to_group: Dict[str, str],
    apd_groups: Dict[str, Dict],
    group_thresh_map: Dict[str, float],
    max_headcap_dist: float = 60.0,
    default_iou_min: float = 0.05,
    center_first: bool = True,
    prefer_ok_over_violation: bool = False,
):
    def area(box):
        x1, y1, x2, y2 = box
        return max(1, x2 - x1) * max(1, y2 - y1)

    apd_by_group = {}
    for d in apd_dets:
        lab = d["label"]
        grp = class_to_group.get(lab)
        if grp is None:
            low = lab.lower()
            if "head" in low or "headcap" in low:
                grp = "headcap"
            elif "mask" in low:
                grp = "mask"
            elif "glove" in low or "gloves" in low:
                grp = "gloves"
            else:
                continue
        apd_by_group.setdefault(grp, []).append(d)

    for grp, dets in apd_by_group.items():
        base_thresh = group_thresh_map.get(grp, default_iou_min)
        for det in dets:
            apd_box = det["bbox"]
            apd_label = det["label"]
            apd_area = area(apd_box)

            best = None
            best_score = -1.0
            best_match_conf = 0.0

            candidates = []
            for tr in active_tracks:
                tr_box = tr["bbox"]
                tr_area = area(tr_box)
                if tr_area <= 0:
                    continue
                center_hit = center_inside(apd_box, tr_box)
                iou = compute_iou(apd_box, tr_box)
                apd_cx, apd_cy = box_center(apd_box)
                px1, py1, px2, py2 = tr_box
                head_dist = abs(apd_cy - py1)
                candidates.append(
                    {
                        "tr": tr,
                        "center": center_hit,
                        "iou": iou,
                        "adapt": base_thresh,
                        "hd": head_dist,
                        "tr_h": (py2 - py1),
                    }
                )

            for c in candidates:
                score = 0.0
                if c["center"] and center_first:
                    score += 100.0
                score += c["iou"] * 10.0
                if grp == "headcap":
                    score -= (c["hd"] / float(max(1, c["tr_h"]))) * 5.0
                if score > best_score:
                    best_score = score
                    best = c
                    match_conf = 0.6 * clamp(c["iou"]) + 0.3 * (1.0 if c["center"] else 0.0)
                    if grp == "headcap":
                        hd_norm = c["hd"] / float(max(1, c["tr_h"]))
                        match_conf += 0.1 * (1.0 - clamp(hd_norm))
                    best_match_conf = clamp(match_conf)

            if best is None:
                continue

            accept = False
            if best["center"] and center_first:
                accept = True
            elif best["iou"] >= best["adapt"]:
                accept = True
            else:
                if grp == "headcap" and best["hd"] <= max_headcap_dist:
                    accept = True

            if not accept:
                continue

            tr = best["tr"]
            apd_conf = det.get("conf", 1.0)
            final_conf = clamp(0.7 * apd_conf + 0.3 * best_match_conf)

            cfg = apd_groups.get(grp, {})
            if apd_label in cfg.get("positive", []):
                typ = "positive"
            elif apd_label in cfg.get("negative", []):
                typ = "negative"
            else:
                low = apd_label.lower()
                typ = "negative" if "no" in low else "positive"

            prev = tr["apd"].get(grp, "unknown")
            if typ == "positive":
                tr["apd"][grp] = "ok"
            elif typ == "negative":
                if prefer_ok_over_violation and prev == "ok":
                    pass
                else:
                    tr["apd"][grp] = "violation"

            if "apd_confidence" not in tr:
                tr["apd_confidence"] = {}

            tr["apd_confidence"][grp] = {
                "confidence": round(final_conf, 3),
                "status": tr["apd"][grp],
            }
            if "apd_bbox" not in tr:
                tr["apd_bbox"] = {}
            tr["apd_bbox"][grp] = apd_box


# ---------------------------------------------------------
# Drawing Overlay Utility (Extracted from apd_utils.py)
# ---------------------------------------------------------
def draw_table_overlay(frame: np.ndarray, tracks: List[dict]) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale_ref = min(fw, fh) / 720.0
    font_scale = max(0.45, min(0.9, 0.5 * scale_ref))
    font_thick = max(1, int(1.6 * scale_ref))
    box_thick = max(1, int(2.2 * scale_ref))
    pad = int(10 * scale_ref)
    row_h = int(25 * scale_ref)

    def text_width(txt):
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        return tw

    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        apd = t["apd"]
        apd_conf = t.get("apd_confidence", {})

        x1 = max(0, min(x1, fw - box_thick - 1))
        y1 = max(0, min(y1, fh - box_thick - 1))
        x2 = max(0, min(x2, fw - box_thick - 1))
        y2 = max(0, min(y2, fh - box_thick - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        ok = all(v == "ok" for v in apd.values())
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)

        rows = [f"ID:{t.get('id','-')}"]
        for g, status in apd.items():
            conf = apd_conf.get(g, {}).get("confidence")
            if conf is not None:
                rows.append(f"{g}: {status} ({conf:.2f})")
            else:
                rows.append(f"{g}: {status}")

        max_panel_w = int(0.35 * fw)
        max_text_w = max(text_width(r) for r in rows)
        w = min(max_text_w + pad * 2, max_panel_w)
        h = row_h * len(rows) + pad

        px = x2 + pad
        if px + w >= fw:
            px = x1 - w - pad
        if px < 0:
            px = x2 + pad

        py = y1
        if py + h >= fh:
            py = y1 - h - pad
        if py < 0:
            py = y2 + pad

        SAFE = max(2, int(3 * scale_ref))
        px = max(SAFE, min(px, fw - w - SAFE))
        py = max(SAFE, min(py, fh - h - SAFE))

        cv2.rectangle(frame, (px, py), (px + w, py + h), (30, 30, 30), -1)

        yy = py + row_h
        for r in rows:
            col = (0, 200, 0) if "ok" in r else ((0, 200, 200) if "unknown" in r else (0, 0, 200))
            cv2.putText(frame, r, (px + pad, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, font_thick, cv2.LINE_AA)
            yy += row_h

    return frame


# ---------------------------------------------------------
# Standalone Local Database (SQLite)
# ---------------------------------------------------------
class LocalViolationDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS violation_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                device_id TEXT,
                camera_id TEXT,
                track_id INTEGER,
                person_bbox TEXT,
                violations TEXT,
                image_path TEXT,
                apd_detail TEXT
            )
            """
            )

    def save_event(self, event_data: dict):
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO violation_events 
                (event_id, timestamp, device_id, camera_id, track_id, person_bbox, violations, image_path, apd_detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event_data["event_id"],
                    event_data["timestamp"],
                    event_data["device_id"],
                    event_data["camera_id"],
                    event_data["track_id"],
                    json.dumps(event_data["person_bbox"]),
                    json.dumps(event_data["violations"]),
                    event_data["image_path"],
                    json.dumps(event_data["apd_detail"]),
                ),
            )


# ---------------------------------------------------------
# Standalone Violation Detection Pipeline Class
# ---------------------------------------------------------
class StandaloneViolationDetector:
    def __init__(
        self,
        output_dir: str = "artifacts/standalone_violations",
        device_id: str = "ED-STANDALONE-01",
        camera_id: str = "CAM01",
        vote_window: int = 5,
        min_age_frames: int = 3,
        min_violation_ratio: float = 0.60,
        cooldown_ms: int = 20000,
    ):
        self.output_dir = output_dir
        self.device_id = device_id
        self.camera_id = camera_id
        self.vote_window = vote_window
        self.min_age_frames = min_age_frames
        self.min_violation_ratio = min_violation_ratio
        self.cooldown_ms = cooldown_ms

        self.snapshots_dir = os.path.join(output_dir, "snapshots", camera_id)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.db = LocalViolationDatabase(os.path.join(output_dir, "violations.db"))
        self.json_log_path = os.path.join(output_dir, "violations.json")

        self.enabled_groups = ["headcap", "mask", "gloves"]
        self.apd_groups = {
            "headcap": {"positive": ["head_cap"], "negative": ["no_head_cap"]},
            "mask": {"positive": ["mask"], "negative": ["no_mask"]},
            "gloves": {"positive": ["gloves"], "negative": ["no_gloves"]},
        }
        self.class_to_group = {
            "head_cap": "headcap",
            "no_head_cap": "headcap",
            "mask": "mask",
            "no_mask": "mask",
            "gloves": "gloves",
            "no_gloves": "gloves",
            "person": "person",
        }
        self.group_thresh = {"headcap": 0.05, "mask": 0.03, "gloves": 0.15}

        # Tracking state
        self.track_state = {}
        self.frame_counter = 0
        self.tracker = BYTETracker(TrackerArgs()) if BYTETracker is not None else None

    def process_frame_detections(
        self,
        frame: np.ndarray,
        person_boxes: List[Tuple[int, int, int, int]],
        apd_dets: List[dict],
        person_confs: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        self.frame_counter += 1
        H, W = frame.shape[:2]

        # Update ByteTrack tracker or fallback to direct indexing
        active_tracks = []
        if self.tracker is not None and len(person_boxes) > 0:
            dets_for_tracker = []
            for i, b in enumerate(person_boxes):
                conf = person_confs[i] if person_confs and i < len(person_confs) else 0.9
                dets_for_tracker.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(conf)])
            dets_arr = np.array(dets_for_tracker, dtype=np.float32)
            try:
                tracks = self.tracker.update(dets_arr, (H, W), (H, W))
                for tr in tracks:
                    tlbr = getattr(tr, "tlbr", None)
                    tid = getattr(tr, "track_id", None) or getattr(tr, "id", None)
                    if tlbr is not None:
                        bbox = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
                        active_tracks.append({"id": int(tid), "bbox": bbox, "apd": {g: "unknown" for g in self.enabled_groups}})
            except Exception as e:
                print(f"[Tracker Warning] {e}")

        if not active_tracks:
            for i, b in enumerate(person_boxes):
                active_tracks.append({"id": i + 1, "bbox": b, "apd": {g: "unknown" for g in self.enabled_groups}})

        # Smart assignment of PPE attribute detections to persons
        assign_apd_smart(
            active_tracks,
            apd_dets,
            self.class_to_group,
            self.apd_groups,
            self.group_thresh,
        )

        # Update rolling vote history & test for violations
        captured_events = []
        now_ms = int(time.time() * 1000)

        for tr in active_tracks:
            tid = tr["id"]
            if tid is None:
                continue

            if tid not in self.track_state:
                self.track_state[tid] = {
                    "first_seen_frame": self.frame_counter,
                    "last_event_ms": 0,
                    "last_event_apd_state": None,
                    "apd_history": {g: deque(maxlen=self.vote_window) for g in self.enabled_groups},
                }
            st = self.track_state[tid]

            for g in self.enabled_groups:
                cur = tr["apd"].get(g, "unknown").lower()
                st["apd_history"][g].append(cur if cur in ("ok", "violation") else "unknown")

            age_frames = self.frame_counter - st["first_seen_frame"]
            if age_frames < self.min_age_frames:
                continue

            voted_violations = []
            for g in self.enabled_groups:
                hist = st["apd_history"][g]
                if not hist:
                    continue
                viol_count = sum(1 for v in hist if v == "violation")
                if (viol_count / float(len(hist))) >= self.min_violation_ratio:
                    voted_violations.append(g)

            if not voted_violations:
                continue

            if st["last_event_apd_state"] == tuple(sorted(voted_violations)):
                continue

            if (now_ms - st["last_event_ms"]) < self.cooldown_ms:
                continue

            # Capture Violation Event!
            event_id = str(uuid.uuid4())
            snapshot_frame = draw_table_overlay(frame.copy(), [tr])
            snapshot_path = os.path.join(self.snapshots_dir, f"{event_id}.jpg")
            cv2.imwrite(snapshot_path, snapshot_frame)

            event_data = {
                "event_id": event_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device_id": self.device_id,
                "camera_id": self.camera_id,
                "track_id": tid,
                "person_bbox": [int(v) for v in tr["bbox"]],
                "violations": voted_violations,
                "image_path": snapshot_path,
                "apd_detail": {g: {"status": tr["apd"].get(g, "unknown")} for g in self.enabled_groups},
            }

            # Save locally to SQLite and JSON log
            self.db.save_event(event_data)
            with open(self.json_log_path, "a") as f:
                f.write(json.dumps(event_data) + "\n")

            st["last_event_apd_state"] = tuple(sorted(voted_violations))
            st["last_event_ms"] = now_ms
            captured_events.append(event_data)

            print(f"🚨 [VIOLATION CAPTURED] ID:{tid} Violations:{voted_violations} Snapshot:{snapshot_path}")

        # Render visual overlay on full frame
        rendered_frame = draw_table_overlay(frame.copy(), active_tracks)
        return rendered_frame, captured_events
