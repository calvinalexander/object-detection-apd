"""Derive apd_monitor_ours.py from the production monitor by four substitutions.

Kept as a patch rather than a fork so it can be re-applied to the live file the
moment it is pulled from the container. Every anchor is asserted: if the live
file differs, this fails loudly instead of silently producing a half-patched
monitor.

What changes (and nothing else):
  1. import our perception module
  2. --backend ultralytics uses a plain capture stream (dev / off-device)
  3. the detection block becomes pose -> ROI -> classifiers
  4. assign_apd_smart(...) becomes assign_ours(...)
  5. CLI gains the model paths; --model defaults to our network name
  6. the event snapshot and stored detail show the vote, not the current frame
Supervisor, ByteTrack, vote window, cooldown, snapshot, SQLite: untouched.
"""
import re, sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
n_before = src.count("\n")

def sub(old, new, what):
    global src
    assert src.count(old) == 1, f"anchor for [{what}] found {src.count(old)}x, expected 1"
    src = src.replace(old, new)

# 1. import
sub("from apd_database import Database\n",
    "from apd_database import Database\n"
    "from apd_perception_ours import OursPerception, assign_ours, LocalStream, voted_view\n", "import")

# 6. the snapshot and stored detail show the vote, not the current frame
sub("                    frame_to_save = draw_table(frame.copy(), [tr])\n",
    "                    # OURS: caption and stored detail show what the vote decided\n"
    "                    # (k of the last n frames), not this single frame\n"
    "                    voted_status, shown = voted_view(tr, st[\"apd_history\"], voted_violations, ENABLED_GROUPS)\n"
    "                    frame_to_save = draw_table(frame.copy(), [shown])\n", "snapshot follows the vote")
sub('                            g: {"status": tr["apd"].get(g, "unknown"), "bbox": tr.get("apd_bbox", {}).get(g, None)}\n',
    '                            g: {"status": voted_status[g], "bbox": tr.get("apd_bbox", {}).get(g, None)}\n',
    "stored detail follows the vote")

# 2. dev stream
sub('''    elif backend == "ultralytics":
        from apd_inference.factory import create_inference_stream
        stream = create_inference_stream(
          backend="ultralytics",
          model_path=args.model,
          sources=args.sources,
          imgsz=args.imgsz,
          device=args.device
    )''',
'''    elif backend == "ultralytics":
        # off-device development: plain capture; pose is run by OursPerception
        stream = LocalStream(args.sources, stride=int(getattr(args, 'stride', 1)))''', "ultralytics stream")

# 3. detection block -> our perception
start = src.index("                meta = frame_result.meta or {}")
end_line = "dets_arr = np.array(dets_for_tracker, dtype=np.float32) if len(dets_for_tracker) > 0 else np.zeros((0,5), dtype=np.float32)"
end = src.index(end_line) + len(end_line)
src = src[:start] + '''                meta = frame_result.meta or {}
                frame = frame_result.image if isinstance(frame_result.image, np.ndarray) else to_numpy(frame_result.image)
                H, W = frame.shape[:2]

                # ---- OUR PERCEPTION: pose -> anatomical ROI -> packed classifiers.
                # Replaces the 7-class detector. Person boxes feed the same tracker;
                # verdicts are attached to tracks after tracking, below.
                _pboxes, _kpts = perception.people(frame, meta, args.model)
                person_boxes, verdicts = perception.perceive(frame, _pboxes, _kpts)
                dets_for_tracker = [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), 0.9] for b in person_boxes]
                dets_arr = np.array(dets_for_tracker, dtype=np.float32) if len(dets_for_tracker) > 0 else np.zeros((0,5), dtype=np.float32)''' + src[end:]

# 4. assignment
sub("assign_apd_smart(active_tracks, apd_dets, class_to_group, APD_GROUPS, APD_GROUP_THRESH)",
    "assign_ours(active_tracks, person_boxes, verdicts)", "assign")

# perception object, created once next to class_to_group
sub("    class_to_group = build_class_maps(CLASS_NAMES, APD_GROUPS)\n",
    "    class_to_group = build_class_maps(CLASS_NAMES, APD_GROUPS)\n"
    "    perception = OursPerception(args)\n"
    "    logger.info(f\"Perception: OURS (pose->ROI->classifiers) model={'packed' if perception.packed is not None else 'two files'} mask={'face-crop trunk' if perception.mask_net is not None else 'packed head trunk'} gloves={'crop-detector' if perception.gloves_det is not None else 'hand-ROI trunk'} thresh={perception.thresh} mask_gate=(nose>={perception.mask_min_nose}, facing={perception.mask_need_facing}) person_conf={perception.conf if perception.pose is not None else 'from config.yaml'}\")\n",
    "perception init")

# 5. CLI
sub('p.add_argument("--model", type=str, default="apd_mbg", help="axelera/ultralytics model")',
    'p.add_argument("--model", type=str, default="apd_ours", help="axelera pose network name")', "model default")
sub("    return p.parse_args()",
'''    # ---- ours
    p.add_argument("--packed_onnx", type=str, default=None, help="all three trunks, one call per frame (default: bundle)")
    p.add_argument("--gloves_onnx", type=str, default=None, help="crop detector that answers gloves without a hand ROI")
    p.add_argument("--mask_onnx", type=str, default=None, help="mask trunk on the face crop (empty = packed head trunk)")
    p.add_argument("--head_onnx", type=str, default="apd_ours_head.onnx", help="fallback: head crop -> head_cap, mask")
    p.add_argument("--hand_onnx", type=str, default="apd_ours_hand.onnx", help="hand crop -> gloves")
    p.add_argument("--pose_weights", type=str, default=None, help="ultralytics backend only")
    p.add_argument("--person_conf", type=float, default=0.40, help="ultralytics backend only; the compiled pipeline uses config.yaml")
    p.add_argument("--gate", type=int, default=1, help="skip ROIs that cannot be read")
    p.add_argument("--stride", type=int, default=1, help="ultralytics backend: process every Nth frame")
    return p.parse_args()''', "cli")

out.write_text(src)
print(f"wrote {out}  ({n_before} -> {src.count(chr(10))} lines)")
