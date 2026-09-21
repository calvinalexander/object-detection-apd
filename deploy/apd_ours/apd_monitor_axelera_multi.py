import argparse, os, time, traceback, json, uuid
from collections import deque, defaultdict
import cv2
import numpy as np
import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from apd_utils import (
    to_int_box, draw_text, draw_table, draw_debug_boxes, to_numpy,
    build_class_maps, assign_apd_smart
)
from apd_database import Database
from apd_perception_ours import OursPerception, assign_ours, LocalStream, voted_view
#from axelera.app import config as ax_config, yaml_parser
#from axelera.app.stream import create_inference_stream
import faulthandler
faulthandler.enable()
# try ByteTrack import (various package layouts)
BYTE_TRACKER_CLASS = None
# original attempt (your project may have tracker.byte_tracker or tracker.bytetrack)
from apd_tracker.byte_tracker import BYTETracker
BYTE_TRACKER_CLASS = BYTETracker

# load config (user editable)
from config_apd import CLASS_NAMES, APD_GROUPS, ENABLED_GROUPS, APD_GROUP_THRESH, DEVICE_ID

from apd_config import config

CLASS_NAMES = config.APD.CLASS_NAMES
APD_GROUPS = config.APD.GROUPS
ENABLED_GROUPS = config.APD.ENABLED_GROUPS
APD_GROUP_THRESH = config.APD.GROUP_THRESH
DEVICE_ID = config.Service.DEVICE_ID
MIN_CONF_PERSON = config.APD.PERSON_THRESH


# =========================
# SETUP LOGGING
# =========================
logger = logging.getLogger("APDMonitor")
logger.setLevel(logging.DEBUG)

log_path = os.path.join(config.Paths.LOG_DIR, "apd_monitor.log")
handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# -----------------------
# Helper: validate sources sebelum dilempar ke stream
# -----------------------
def validate_sources(sources: list, cam_ids=None):
    """
    Cek setiap source apakah bisa dibuka. Source yang gagal akan dilewati.
    Return: (valid_sources, valid_cam_ids)
    """
    valid_sources = []
    valid_cam_ids = []
    for i, src in enumerate(sources):
        cam_label = cam_ids[i] if cam_ids and i < len(cam_ids) else f"CAM{i:02d}"
        try:
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                valid_sources.append(src)
                valid_cam_ids.append(cam_label)
                logger.info(f"Source OK [{cam_label}]: {src}")
            else:
                logger.error(f"Source GAGAL dibuka [{cam_label}] (dilewati): {src}")
            cap.release()
        except Exception as e:
            logger.error(f"Source error [{cam_label}] (dilewati): {src} -> {e}")
    return valid_sources, valid_cam_ids

# -----------------------
# Helper: create tracker factory (adjust if your tracker requires different args)
# -----------------------
def create_tracker_instance(args):
    """
    Return a new tracker instance. If BYTE_TRACKER_CLASS requires different constructor args,
    adjust this function accordingly.
    """
    if BYTE_TRACKER_CLASS is None:
        return None
    try:
        # Many ByteTrack implementations accept an `args` object or config dict.
        return BYTE_TRACKER_CLASS(args)
    except Exception:
        # fallback to no-arg constructor if available
        try:
            return BYTE_TRACKER_CLASS()
        except Exception:
            raise RuntimeError("Cannot instantiate BYTETracker: adapt create_tracker_instance() to your tracker signature.")

# -----------------------
# Main loop
# -----------------------
def apd_monitor_main(args):
    backend=args.backend

    # -----------------------------------------
    # Validasi sources sebelum stream dibuat
    # -----------------------------------------
    valid_sources, valid_cam_ids = validate_sources(args.sources, args.cam_ids)
    if not valid_sources:
        raise RuntimeError("Tidak ada source CCTV yang valid. Periksa konfigurasi sources!")
    if len(valid_sources) < len(args.sources):
        logger.warning(
            f"{len(args.sources) - len(valid_sources)} source dilewati karena gagal. "
            f"Melanjutkan dengan {len(valid_sources)} source yang valid."
        )
    args.sources = valid_sources
    args.cam_ids = valid_cam_ids

    stream = None
    if backend == "axelera":
        from axelera.app import config as ax_config
        from axelera.app.stream import create_inference_stream
        stream = create_inference_stream(
            network=args.model,
            sources=args.sources,
            pipe_type=args.pipe,
            #log_level=logging_utils.INFO,  # INFO, DEBUG, TRACE
            hardware_caps=ax_config.HardwareCaps(
                vaapi=ax_config.HardwareEnable.detect,
                opencl=ax_config.HardwareEnable.detect,
                opengl=ax_config.HardwareEnable.detect,
            ),
            aipu_cores=2,
            specified_frame_rate=10,
            # rtsp_latency=500,
    )
        
    elif backend == "ultralytics":
        # off-device development: plain capture; pose is run by OursPerception
        stream = LocalStream(args.sources, stride=int(getattr(args, 'stride', 1)))
       
    if stream is None:
        raise RuntimeError("Failed to create Inference Stream, make sure configuration is correct !")

    # -----------------------------------------
    # Build CAM_ID MAP (stream_id -> cam_id)
    # -----------------------------------------
    cam_id_map = {}

    if args.cam_ids:
        # contoh: --sources s1 s2 s3  → stream_id = 0,1,2
        #         --cam_ids CAM01 CAM02 CAM03
        for i, cam in enumerate(args.cam_ids):
            cam_id_map[i] = cam
    else:
        # fallback: generate CAM00, CAM01...
        for i in range(len(args.sources)):
            cam_id_map[i] = f"CAM{i:02d}"

    logger.info(f"CAM ID mapping: {cam_id_map}")

    class_to_group = build_class_maps(CLASS_NAMES, APD_GROUPS)
    perception = OursPerception(args)
    logger.info(f"Perception: OURS (pose->ROI->classifiers) model={'packed' if perception.packed is not None else 'two files'} mask={'face-crop trunk' if perception.mask_net is not None else 'packed head trunk'} gloves={'crop-detector' if perception.gloves_det is not None else 'hand-ROI trunk'} thresh={perception.thresh} mask_gate=(nose>={perception.mask_min_nose}, facing={perception.mask_need_facing}) person_conf={perception.conf if perception.pose is not None else 'from config.yaml'}")

    # ---------------------------------------
    # MULTI-STREAM STATE (per stream_id)
    # ---------------------------------------
    FRAME_ID       = defaultdict(int)            # FRAME_ID[sid] -> counter
    TRACK_STATE    = defaultdict(dict)           # TRACK_STATE[sid][track_id] = {...}
    RECENT_EVENTS  = defaultdict(dict)           # RECENT_EVENTS[sid][track_id] = last_ms
    FPS_TIME       = defaultdict(float)          # FPS_TIME[sid] -> prev timestamp
    WRITER         = defaultdict(lambda: None)   # WRITER[sid] -> VideoWriter or None

    # Trackers per stream
    TRACKERS = defaultdict(lambda: create_tracker_instance(args))

    # Ensure tracker class is available
    if BYTE_TRACKER_CLASS is None:
        raise RuntimeError("ByteTrack class not found. Install a ByteTrack implementation or adapt create_tracker_instance().")

    os.makedirs(args.outdir, exist_ok=True)
    logger.info("Starting APD monitor (multi-stream + per-stream tracker); press q to quit")

    db = Database()
    db.init_db()

    # Voting / thresholds (shared)
    VOTE_WINDOW = config.APD.VOTE_WINDOW
    MIN_AGE_FRAMES = config.APD.MIN_AGE_FRAMES
    MIN_VIOLATION_RATIO = config.APD.MIN_VIOLATION_RATIO
    COOLDOWN_MS = config.APD.COOLDOWN_MS

    session_id = str(uuid.uuid4())

    try:
        for frame_result in stream:
            # === per-stream bookkeeping ===
            sid = getattr(frame_result, "stream_id", 0)
            try:
                FRAME_ID[sid] += 1
                frame_id = FRAME_ID[sid]

                if frame_result.image is None:
                    logger.warning(f"WARNING: Null frame for stream {sid} at frame {frame_id}")
                    continue

                meta = frame_result.meta or {}
                frame = frame_result.image if isinstance(frame_result.image, np.ndarray) else to_numpy(frame_result.image)
                H, W = frame.shape[:2]

                # ---- OUR PERCEPTION: pose -> anatomical ROI -> packed classifiers.
                # Replaces the 7-class detector. Person boxes feed the same tracker;
                # verdicts are attached to tracks after tracking, below.
                _pboxes, _kpts = perception.people(frame, meta, args.model)
                person_boxes, verdicts = perception.perceive(frame, _pboxes, _kpts)
                dets_for_tracker = [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), 0.9] for b in person_boxes]
                dets_arr = np.array(dets_for_tracker, dtype=np.float32) if len(dets_for_tracker) > 0 else np.zeros((0,5), dtype=np.float32)

                # --- per-stream tracker usage ---
                tracker = TRACKERS[sid]
                tracks = None
                if tracker is not None:
                    try:
                        # keep original signature: update(dets, frame_size, ???)
                        # many implementations used: tracker.update(dets_arr, (H,W), (H,W))
                        # if your tracker signature differs, adapt here.
                        tracks = tracker.update(dets_arr, (H,W), (H,W))
                    except Exception as e:
                        logger.error(f"[stream {sid}] Tracker.update exception: {e}")
                        tracks = None
                else:
                    tracks = None

                # build active_tracks list
                active_tracks = []
                if tracks is None:
                    for i, b in enumerate(person_boxes):
                        active_tracks.append({
                            "id": i+1,
                            "bbox": b,
                            "apd": {g: "unknown" for g in ENABLED_GROUPS}
                        })
                else:
                    for tr in tracks:
                        tlbr = getattr(tr, "tlbr", None)
                        tid = getattr(tr, "track_id", None) or getattr(tr, "trackId", None) or getattr(tr, "id", None)
                        if tlbr is None:
                            continue
                        bbox = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
                        active_tracks.append({
                            "id": int(tid) if tid is not None else -1,
                            "bbox": bbox,
                            "apd": {g: "unknown" for g in ENABLED_GROUPS}
                        })

                # assign APD to active_tracks (in-place)
                assign_ours(active_tracks, person_boxes, verdicts)

                # --- update per-track state storage (per stream) ---
                for tr in active_tracks:
                    tid = tr["id"]
                    if tid is None:
                        continue

                    st = TRACK_STATE[sid].get(tid)
                    if st is None:
                        st = {
                            "first_seen_frame": frame_id,
                            "last_event_ms": 0,
                            "apd_history": {g: deque(maxlen=VOTE_WINDOW) for g in ENABLED_GROUPS}
                        }
                        TRACK_STATE[sid][tid] = st

                    # append current apd status
                    for g in ENABLED_GROUPS:
                        cur = tr["apd"].get(g, "unknown").lower()
                        if cur not in ("ok", "violation"):
                            cur = "unknown"
                        st["apd_history"][g].append(cur)

                # -----------------------
                # Save event ke database (per stream)
                # -----------------------

                for tr in active_tracks:
                    track_id = tr.get("id", None)
                    if track_id is None:
                        continue

                    st = TRACK_STATE[sid].get(track_id)
                    if st is None:
                        continue

                    age_frames = frame_id - st.get("first_seen_frame", frame_id)
                    if age_frames < MIN_AGE_FRAMES:
                        continue

                    # compute voting (percentage-based)
                    voted_violations = []
                    for g in ENABLED_GROUPS:
                        hist = st["apd_history"].get(g, deque())
                        if len(hist) == 0:
                            continue
                        viol_count = sum(1 for v in hist if v == "violation")
                        ratio = viol_count / float(len(hist))
                        if ratio >= MIN_VIOLATION_RATIO:
                            voted_violations.append(g)

                    if not voted_violations:
                        continue

                    last_state = st.get("last_event_apd_state")
                    if last_state is not None and last_state == tuple(sorted(voted_violations)):
                        continue

                    now_ms = int(time.time() * 1000)
                    if now_ms - st.get("last_event_ms", 0) < COOLDOWN_MS:
                        continue

                    event_id = str(uuid.uuid4())

                    # snapshot per event (1 event_id = 1 image)
                    # OURS: caption and stored detail show what the vote decided
                    # (k of the last n frames), not this single frame
                    voted_status, shown = voted_view(tr, st["apd_history"], voted_violations, ENABLED_GROUPS)
                    frame_to_save = draw_table(frame.copy(), [shown])
                    #cam_dir = os.path.join(args.outdir, f"cam_{sid}")
                    cam_dir = os.path.join(args.outdir, cam_id_map.get(sid, f"cam_{sid}"))
                    os.makedirs(cam_dir, exist_ok=True)
                    fname = f"{event_id}.jpg"
                    frame_saved_path = os.path.join(cam_dir, fname)
                    cv2.imwrite(frame_saved_path, frame_to_save)

                    # build event payload
                    event_json = {
                        "event_id": event_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "device_id": DEVICE_ID,
                        "camera_id": cam_id_map.get(sid, f"CAM{sid:02d}"),
                        #"camera_id": f'CAM{sid:02d}',
                        "session_id": session_id,
                        "track_id": track_id,
                        "person_bbox": [max(1, int(v)) for v in tr["bbox"]],
                        "violation": json.dumps(voted_violations),
                        "image_path": frame_saved_path,
                        "apd_detail": {
                            g: {"status": voted_status[g], "bbox": tr.get("apd_bbox", {}).get(g, None)}
                            for g in ENABLED_GROUPS
                        }
                    }

                    try:
                        db.save_event(event_json)
                    except Exception as e:
                        logger.error(f"[stream {sid}] DB save_event exception: {e}")

                    # update state + recent map
                    st["last_event_apd_state"] = tuple(sorted(voted_violations))
                    st["last_event_ms"] = now_ms
                    RECENT_EVENTS[sid][track_id] = now_ms

                # -----------------------
                # Draw & optional display/save
                # -----------------------
                if args.mode == "text":
                    out = draw_text(frame, active_tracks)
                elif args.mode == "table":
                    out = draw_table(frame, active_tracks)
                else:
                    out = draw_text(frame, active_tracks)

                # ====== hitung fps per stream =====
                current_time = time.time()
                prev = FPS_TIME[sid]
                fps = 1 / (current_time - prev) if prev != 0 else 0
                FPS_TIME[sid] = current_time
                if args.show_fps:
                    cv2.putText(out, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                # save per-stream writer
                if args.save:
                    if WRITER[sid] is None:
                        outpath = args.outpath
                        if outpath.endswith(".mp4"):
                            #outpath = outpath.replace(".mp4", f"_cam{sid}.mp4")
                            outpath = outpath.replace(
                                            ".mp4",
                                            f"_{cam_id_map.get(sid, f'cam{sid}')}.mp4"
                                        )
                        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
                        WRITER[sid] = cv2.VideoWriter(
                            outpath,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps if args.fps > 0 else 20.0,
                            (out.shape[1], out.shape[0])
                        )
                    try:
                        WRITER[sid].write(out)
                    except Exception as e:
                        logger.error(f"[stream {sid}] Error writing frame: {e}")

                # show per-stream window
                if args.display:
                    win_name = f"APD Monitor - {cam_id_map.get(sid, 'UNKNOWN')}"
                    #win_name = f"APD Monitor Axelera - CAM {sid}"
                    cv2.imshow(win_name, out)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        # user requested quit -> break outer loop
                        raise KeyboardInterrupt()

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"[stream {sid}] Frame processing error, skipping frame: {e}", exc_info=True)
                continue

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user / quit key.")
    except Exception as e:
        logger.error(f"\nFatal exception in stream iterator: {e}")
        traceback.print_exc()
    finally:
        try:
            stream.stop()
        except Exception:
            pass

        # release all writers
        for sid_k, w in WRITER.items():
            try:
                if w is not None:
                    w.release()
            except Exception:
                pass

        cv2.destroyAllWindows()
        try:
            db.close()
        except Exception:
            pass

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="apd_ours", help="axelera pose network name")
    p.add_argument("--sources", nargs="+", required=True, help="Daftar input video / RTSP")
    p.add_argument("--cam_ids", nargs="+", required=False, help="Daftar cam_id sesuai urutan sources")
    p.add_argument("--mode", type=str, default="text", choices=["text","table","icon"])
    p.add_argument("--pipe", type=str, default="torch-aipu", choices=["torch-aipu","torch","gst"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save", action="store_true", help="save output video(s)")
    p.add_argument("--display", action="store_true", help="show cv imshow windows")
    p.add_argument("--outdir", type=str, default="apd_storage/snapshots")
    p.add_argument("--outpath", type=str, default="out.mp4")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--show_fps", action="store_true", help="show fps on frame")
    p.add_argument("--backend", type=str, default="axelera", choices=["axelera","ultralytics"])
    p.add_argument("--device", type=str, default="intel:gpu", choices=["intel:gpu","intel:cpu","intel:xpu", "cpu", "cuda", "0", "0,1"])

    # tracking params (keperluan tracker internal - mungkin dipakai oleh konstruktor tracker)
    p.add_argument("--track_thresh", type=float, default=0.30, help="tracking confidence threshold")
    p.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    p.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    p.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    p.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    p.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # ---- ours
    p.add_argument("--packed_onnx", type=str, default=None, help="all three trunks, one call per frame (default: bundle)")
    p.add_argument("--gloves_onnx", type=str, default=None, help="crop detector that answers gloves without a hand ROI")
    p.add_argument("--mask_onnx", type=str, default=None, help="mask trunk on the face crop (empty = packed head trunk)")
    p.add_argument("--head_onnx", type=str, default="apd_ours_head.onnx", help="fallback: head crop -> head_cap, mask")
    p.add_argument("--hand_onnx", type=str, default="apd_ours_hand.onnx", help="hand crop -> gloves")
    p.add_argument("--pose_weights", type=str, default=None, help="ultralytics backend only")
    p.add_argument("--person_conf", type=float, default=0.40, help="ultralytics backend only; the compiled pipeline uses config.yaml")
    p.add_argument("--gate", type=int, default=1, help="skip ROIs that cannot be read")
    p.add_argument("--stride", type=int, default=1, help="ultralytics backend: process every Nth frame")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        apd_monitor_main(args)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt, exiting")
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
