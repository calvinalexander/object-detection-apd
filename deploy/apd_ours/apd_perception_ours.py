"""Our perception stage, shaped to drop into the production monitor unchanged.

The production loop is: stream -> detections -> person boxes -> ByteTrack ->
assign PPE verdicts onto tracks -> vote window -> snapshot + DB. Everything
after "person boxes" is kept exactly as it is. This module replaces only the
first two steps:

    apd_v3 (7-class detector) + assign_apd_smart(...)      ← production
    pose -> anatomical ROI -> packed classifiers + assign_ours(...)   ← here

The contract at the seam is the one the monitor already relies on: each active
track gets tr["apd"][group] in {"ok","violation","unknown"}, plus
tr["apd_confidence"] and tr["apd_bbox"] so draw_table and apd_detail work.
Group names are the monitor's ("headcap","mask","gloves"), not ours.

Two pose sources, because this must be testable off the device:
    axelera      keypoints come from the stream meta (CocoBodyKeypointsMeta)
    ultralytics  a YOLO pose model runs here, on a plain cv2 capture
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import cv2
import numpy as np

# apd/ lives two levels up from deploy/apd_ours/ in our repo; on the device it
# is copied next to this file.
for _p in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parent):
    if (_p / "apd" / "pose.py").exists():
        sys.path.insert(0, str(_p)); break
from apd.pose import build_rois, roi_usable
from apd.schema import GEOMETRY

GROUP_OF = {"head_cap": "headcap", "mask": "mask", "gloves": "gloves"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(1e-6, u)


def _crop(img, box, size):
    x1, y1, x2, y2 = box
    pad = 0.35 * max(x2 - x1, y2 - y1)
    c = img[max(0, int(y1-pad)):int(y2+pad), max(0, int(x1-pad)):int(x2+pad)]
    return None if c.size == 0 or min(c.shape[:2]) < 8 else cv2.resize(c, (size, size))


def _prep1(c):
    """One crop -> CHW float tensor, the normalisation the trunks were trained with."""
    x = (cv2.cvtColor(cv2.resize(c, (128, 128)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - MEAN)/STD
    return x.transpose(2, 0, 1)


def _prep(c):
    return _prep1(c)[None]


DET_NAMES = ["person", "gloves", "head_cap", "mask", "no_gloves", "no_head_cap", "no_mask"]


def _letterbox(im, s):
    h, w = im.shape[:2]; r = min(s/h, s/w); nh, nw = max(1, round(h*r)), max(1, round(w*r))
    o = np.full((s, s, 3), 114, np.uint8); t, l = (s-nh)//2, (s-nw)//2
    o[t:t+nh, l:l+nw] = cv2.resize(im, (nw, nh))
    return (cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32)/255).transpose(2, 0, 1)[None], r, l, t


def _stack(crops):
    return (np.stack([_prep1(c) for c in crops]) if crops
            else np.zeros((0, 3, 128, 128), np.float32))


class LocalStream:
    """Stand-in for create_inference_stream() when developing without the AIPU.

    Yields objects with .image / .meta / .stream_id like the real stream does,
    so the monitor's loop body is identical. meta is empty: pose is computed by
    OursPerception itself in that mode.
    """
    class _Frame:
        def __init__(self, image, sid): self.image, self.meta, self.stream_id = image, {}, sid
    def __init__(self, sources, stride=1):
        self.caps = [cv2.VideoCapture(s) for s in sources]; self.stride = max(1, stride); self.n = 0
    def __iter__(self):
        while True:
            alive = False
            for sid, cap in enumerate(self.caps):
                ok, f = cap.read()
                if not ok: continue
                alive = True; self.n += 1
                if self.n % self.stride: continue
                yield LocalStream._Frame(f, sid)
            if not alive: return
    def stop(self):
        for c in self.caps: c.release()


class OursPerception:
    def __init__(self, args):
        import onnxruntime as ort
        opts = ort.SessionOptions(); opts.log_severity_level = 3
        # One frame's crops go through in one call, so the threads that matter
        # are the intra-op ones. Four oversubscribe a 4-core host once the pose
        # stage and the decoder are also running; two measured fastest.
        opts.intra_op_num_threads = int(os.environ.get("APD_OURS_THREADS", 2))
        opts.inter_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        # Under the production supervisor, args is a SimpleNamespace that knows
        # nothing of these options; resolve them relative to this file so the
        # bundle works from whatever working_dir the container uses.
        here = Path(__file__).resolve().parent
        packed = getattr(args, "packed_onnx", None) or os.environ.get("APD_OURS_PACKED_ONNX") or str(here / "apd_ours_packed.onnx")
        self.head = self.hand = self.packed = None
        if Path(packed).exists():
            # The packed graph holds the same three independent trunks, and
            # takes every head crop and every hand crop of a frame as two
            # batches -> one call per frame instead of 3N. Bit-identical
            # outputs (asserted in scripts/86), measurably faster: 64 ms vs
            # 104 ms for three people, and the gap widens with thread count.
            self.packed = ort.InferenceSession(packed, opts, providers=prov)
        else:
            head = getattr(args, "head_onnx", None) or os.environ.get("APD_OURS_HEAD_ONNX") or str(here / "apd_ours_head.onnx")
            hand = getattr(args, "hand_onnx", None) or os.environ.get("APD_OURS_HAND_ONNX") or str(here / "apd_ours_hand.onnx")
            self.head = ort.InferenceSession(head, opts, providers=prov)
            self.hand = ort.InferenceSession(hand, opts, providers=prov)
        self.gate = bool(int(getattr(args, "gate", os.environ.get("APD_OURS_GATE", 1))))

        # Gloves only. The hand ROI is built from wrist keypoints, the least
        # reliable ones: measured against the ROIs these trunks were trained on
        # it agrees only 0.48 median, wrong in 56% of cases, and gloves MCC
        # falls from 0.410 on a correct ROI to 0.099 on a pose one. A crop
        # detector needs no hand ROI at all -- it searches the whole person --
        # which is why it scores 0.156 where the trunk scores below zero.
        # shipped default; APD_OURS_GLOVES_ONNX="" disables it (an empty value
        # must not fall through to the default, so test for None, not falsiness)
        gl = (getattr(args, "gloves_onnx", None) if getattr(args, "gloves_onnx", None) is not None
              else os.environ.get("APD_OURS_GLOVES_ONNX", str(here / "apd_ours_gloves.onnx")))
        if gl and not Path(gl).is_absolute(): gl = str(here / gl)
        self.gloves_det = ort.InferenceSession(gl, opts, providers=prov) if gl and Path(gl).exists() else None
        # Per-attribute cut on the trunk probability that the item is PRESENT.
        # Mask: 0.60, measured on 660 annotator-labelled faces (305 mask on) from
        # production's validation set that the trunk never saw (MCC 0.590 at 0.5,
        # 0.609 at 0.6, 0.548 at 0.8). An earlier 0.80 came from the COCO labels,
        # which hold only 6 people with a mask on; at 0.80 it flags 41% of the
        # people who ARE wearing one. See artifacts/error_analysis/mask_sdk/.
        # Dedicated mask trunk on its own face crop (apd.pose.face_roi, 0.8x
        # shoulder width, centred on the nose, pushed toward the mouth). On 323
        # held-out faces it scores MCC 0.826 against 0.635 for the packed head
        # trunk, flags 14 mask-wearers instead of 27, and holds 0.795 under 10%
        # keypoint noise. Set APD_OURS_MASK_ONNX="" to fall back to the packed
        # trunk. Its threshold (0.30) was chosen on validation, not on test.
        mk = (getattr(args, "mask_onnx", None) if getattr(args, "mask_onnx", None) is not None
              else os.environ.get("APD_OURS_MASK_ONNX", str(here / "apd_ours_mask.onnx")))
        if mk and not Path(mk).is_absolute(): mk = str(here / mk)
        self.mask_net = ort.InferenceSession(mk, opts, providers=prov) if mk and Path(mk).exists() else None
        self.thresh = {"headcap": float(os.environ.get("APD_OURS_HEADCAP_THRESH", 0.5)),
                       "mask": float(os.environ.get("APD_OURS_MASK_THRESH",
                                                    0.30 if self.mask_net is not None else 0.6)),
                       "gloves": float(os.environ.get("APD_OURS_GLOVES_THRESH", 0.5))}
        # Facing gate for mask. The mouth has to be visible to judge a mask, and
        # the head gate alone let 91% of the people a human marked "mouth not
        # visible" through -- mostly backs of heads. Two cues from the pose
        # keypoints: nose confidence (median 0.30 on those people against 0.93
        # on judgeable faces) and left/right order, since facing the camera a
        # person's left shoulder and ear appear on the image's right. Together
        # they refuse 73% of the unjudgeable people and keep 89% of the
        # judgeable ones (scripts/104_mask_facing_gate.py).
        self.mask_min_nose = float(os.environ.get("APD_OURS_MASK_MIN_NOSE", 0.5))
        self.mask_need_facing = bool(int(os.environ.get("APD_OURS_MASK_REQUIRE_FACING", 1)))
        self.det_imgsz = int(os.environ.get("APD_OURS_DET_IMGSZ", 416))
        self.det_conf = float(os.environ.get("APD_OURS_DET_CONF", 0.25))
        self.det_pad = float(os.environ.get("APD_OURS_DET_PAD", 0.18))
        self.pose = None
        if getattr(args, "backend", "axelera") == "ultralytics":
            from ultralytics import YOLO
            self.pose = YOLO(getattr(args, "pose_weights", None) or str(here / "yolo11s-pose.pt"))
            self.imgsz = int(getattr(args, "imgsz", 640))
            # NOT APD_THRESH_PERSON: that is production's threshold on its
            # detector's person class, and it is 0.7 live. The pose network is
            # a different model with a different score distribution -- at 0.7 it
            # drops 90% of the people in our recordings. The compiled pipeline
            # uses conf_threshold 0.40 in config.yaml, so this path matches it.
            self.conf = float(getattr(args, "person_conf", None) or
                              os.environ.get("APD_OURS_PERSON_CONF", 0.40))

    # ---- pose ---------------------------------------------------------------
    def people(self, frame, meta, model_name):
        """(boxes[N,4], keypoints[N,17,3]) from the stream meta, or from local pose."""
        if self.pose is not None:
            r = self.pose.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
            if r.keypoints is None or not len(r.boxes):
                return np.zeros((0, 4)), np.zeros((0, 17, 3))
            return r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()
        m = meta.get(model_name) if hasattr(meta, "get") else getattr(meta, model_name, None)
        if m is None and hasattr(meta, "get"):
            m = meta.get("person_pose") or meta.get("keypoints")
        if m is None or not hasattr(m, "boxes"):
            return np.zeros((0, 4)), np.zeros((0, 17, 3))
        boxes = np.asarray(m.boxes, dtype=np.float32).reshape(-1, 4)
        k = np.asarray(m.keypoints, dtype=np.float32)
        if k.ndim == 3 and k.shape[-1] == 2:                       # no per-point score
            k = np.concatenate([k, np.ones((*k.shape[:2], 1), np.float32)], axis=-1)
        return boxes, k.reshape(-1, 17, 3)

    def _gloves_from_crop(self, frame, person_box):
        return _detector_gloves(self.gloves_det, frame, person_box,
                                self.det_imgsz, self.det_conf, self.det_pad)

    # ---- classify -----------------------------------------------------------
    def _run(self, head_crops, hand_crops):
        """(head_cap[N], mask[N], gloves[M]) for a whole frame, in one call."""
        if self.packed is not None:
            return self.packed.run(None, {"head_crops": _stack(head_crops),
                                          "hand_crops": _stack(hand_crops)})
        # fallback: the per-ROI files, which are fixed batch 1
        h = [self.head.run(None, {"input": _prep(c)})[0][0] for c in head_crops]
        g = [self.hand.run(None, {"input": _prep(c)})[0][0][0] for c in hand_crops]
        return (np.array([x[0] for x in h], np.float32),
                np.array([x[1] for x in h], np.float32),
                np.array(g, np.float32))

    def perceive(self, frame, boxes, kpts):
        """Per person: box, verdict dict in the monitor's vocabulary, confidences, roi boxes.

        Two passes over the people, with one inference between them. The crops
        of every person in the frame are classified together, because the three
        trunks are one graph and a batch of eight heads costs far less than
        eight batches of one.
        """
        if len(boxes) == 0:
            return [], []
        people = build_rois(np.asarray(boxes), np.asarray(kpts))
        pboxes = [p.person_box for p in people]

        # pass 1 -- decide which ROIs are usable, crop them, remember the rows
        head_crops, hand_crops, face_crops, slots = [], [], [], []
        kps = np.asarray(kpts)
        for pi, pp in enumerate(people):
            facing = _facing_camera(kps[pi], self.mask_min_nose, self.mask_need_facing)
            head_ok = (pp.head is not None and not pp.head_collision and
                       (not self.gate or roi_usable(pp.head, frame.shape, part="head",
                                                    others=pboxes, self_box=pp.person_box)))
            hc = _crop(frame, pp.head, GEOMETRY.head_crop_px) if head_ok else None
            hrow = -1
            if hc is not None:
                hrow = len(head_crops); head_crops.append(hc)
            # same admissibility as the head: a face we would not judge for a
            # cap is not one we judge for a mask either
            frow = -1
            if self.mask_net is not None and hc is not None and pp.face is not None and facing:
                fc = _crop(frame, pp.face, GEOMETRY.face_crop_px)
                if fc is not None:
                    frow = len(face_crops); face_crops.append(fc)
            hands, hboxes = [], []
            for h in pp.hands[:2]:
                if self.gate and not roi_usable(h, frame.shape, part="hand", others=pboxes, self_box=pp.person_box):
                    continue
                c = _crop(frame, h, GEOMETRY.hand_crop_px)
                if c is not None: hands.append(c); hboxes.append(h)
            drows = []
            if len(hands) >= 2:                                # BOTH hands must be seen
                for c in hands[:2]:
                    drows.append(len(hand_crops)); hand_crops.append(c)
            slots.append((pp, hrow, frow, drows, hboxes[:2], facing))

        # one inference for the whole frame
        p_cap, p_mask, p_gl = self._run(head_crops, hand_crops)
        p_face = (self.mask_net.run(None, {"face_crops": _stack(face_crops)})[0]
                  if self.mask_net is not None and face_crops else None)

        # pass 2 -- turn probabilities into the monitor's vocabulary
        person_boxes, verdicts = [], []
        for pp, hrow, frow, drows, hboxes, facing in slots:
            v, conf, rb = {}, {}, {}
            if hrow < 0:
                v["headcap"] = v["mask"] = "unknown"
            else:
                judged = [("headcap", float(p_cap[hrow]), pp.head)]
                if not facing:
                    v["mask"] = "unknown"                 # mouth not visible: no call
                elif self.mask_net is None:
                    judged.append(("mask", float(p_mask[hrow]), pp.head))
                elif frow >= 0:
                    judged.append(("mask", float(p_face[frow]), pp.face))
                else:
                    v["mask"] = "unknown"
                for grp, prob, roi in judged:
                    th = self.thresh[grp]
                    v[grp] = "ok" if prob >= th else "violation"
                    conf[grp] = prob if prob >= th else 1.0 - prob
                    rb[grp] = [int(x) for x in roi]
            if self.gloves_det is not None:
                v["gloves"], c, bb = self._gloves_from_crop(frame, pp.person_box)
                if c is not None: conf["gloves"] = c
                if bb is not None: rb["gloves"] = bb
            elif len(drows) < 2:
                v["gloves"] = "unknown"
            else:
                prob = min(float(p_gl[r]) for r in drows)      # BOTH hands must be gloved
                th = self.thresh["gloves"]
                v["gloves"] = "ok" if prob >= th else "violation"
                conf["gloves"] = prob if prob >= th else 1.0 - prob
                x1 = min(b[0] for b in hboxes); y1 = min(b[1] for b in hboxes)
                x2 = max(b[2] for b in hboxes); y2 = max(b[3] for b in hboxes)
                rb["gloves"] = [int(x1), int(y1), int(x2), int(y2)]
            person_boxes.append(tuple(int(x) for x in pp.person_box))
            verdicts.append({"apd": v, "conf": conf, "bbox": rb})
        return person_boxes, verdicts


def _facing_camera(kp, min_nose, need_facing, vis=0.35):
    """Can this person's mouth be seen? False for backs of heads."""
    if kp[0][2] < min_nose:
        return False
    if need_facing:
        for l, r in ((5, 6), (3, 4)):             # shoulders, ears
            if kp[l][2] >= vis and kp[r][2] >= vis and kp[l][0] <= kp[r][0]:
                return False                      # left side on the image's left: turned away
    return True


def _detector_gloves(sess, frame, person_box, imgsz, conf_th, pad):
    """gloves / no_gloves for one person, read off a crop detector.

    No hand ROI is involved, so a bad wrist keypoint cannot mislead it. The
    export is NMS-free (YOLO26), so the output is already the final boxes.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = person_box
    pw, ph = x2-x1, y2-y1
    cx1, cy1 = int(max(0, x1-pad*pw)), int(max(0, y1-pad*ph))
    cx2, cy2 = int(min(W, x2+pad*pw)), int(min(H, y2+pad*ph))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return "unknown", None, None
    x, r, dx, dy = _letterbox(crop, imgsz)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    best = {}
    for row in out:
        sc, cl = float(row[4]), int(row[5])
        if sc < conf_th: continue
        nm = DET_NAMES[cl] if 0 <= cl < len(DET_NAMES) else None
        if nm in ("gloves", "no_gloves") and sc > best.get(nm, (0, None))[0]:
            b = [(row[0]-dx)/r + cx1, (row[1]-dy)/r + cy1, (row[2]-dx)/r + cx1, (row[3]-dy)/r + cy1]
            best[nm] = (sc, [int(v) for v in b])
    pos, neg = best.get("gloves"), best.get("no_gloves")
    if pos and neg:                       # both seen: the more confident one wins
        return ("ok", pos[0], pos[1]) if pos[0] >= neg[0] else ("violation", neg[0], neg[1])
    if pos: return "ok", pos[0], pos[1]
    if neg: return "violation", neg[0], neg[1]
    return "unknown", None, None


def voted_view(tr, history, voted, groups):
    """What an event actually decided, per group, from the vote window.

    Production draws the snapshot from the current frame, but decides the event
    from the last n frames, so a snapshot could read "mask: ok" on a mask alert.
    Here both the caption and the stored detail show the vote:

        violation  the group is in the voted list (>= MIN_VIOLATION_RATIO of the window)
        ok         otherwise, if "ok" frames are at least as many as "violation" ones
        unknown    otherwise (mostly no call: facing away, head unusable, ...)

    The caption reads "mask: violation 4/5" -- 4 of the last 5 frames. draw_table
    colours a row green when it contains "ok", yellow for "unknown", red otherwise,
    so the colours follow the same decision.
    Returns (plain statuses for the database, a track dict for draw_table).
    """
    status, shown = {}, {}
    for g in groups:
        h = list(history.get(g, [])); n = len(h)
        nv, no = h.count("violation"), h.count("ok")
        if g in voted: st, k = "violation", nv
        elif no and no >= nv: st, k = "ok", no
        else: st, k = "unknown", h.count("unknown")
        status[g] = st
        shown[g] = f"{st} {k}/{n}" if n else st
    view = dict(tr); view["apd"] = shown; view.pop("apd_confidence", None)
    return status, view


def assign_ours(active_tracks, person_boxes, verdicts, min_iou=0.3):
    """Put each pose-person's verdict onto the ByteTrack track it belongs to.

    Replaces assign_apd_smart(). That function had to guess which person a PPE
    box belonged to; here the verdict was computed per person already, so the
    only question is which track is that person -- answered by IoU against the
    tracked box. An unmatched track keeps "unknown", which the vote counts in
    the denominator only: silence never accuses.
    """
    used = set()
    for tr in active_tracks:
        best, bi = 0.0, -1
        for i, pb in enumerate(person_boxes):
            if i in used: continue
            s = _iou(tr["bbox"], pb)
            if s > best: best, bi = s, i
        if bi < 0 or best < min_iou:
            continue
        used.add(bi)
        vd = verdicts[bi]
        for grp, st in vd["apd"].items():
            tr["apd"][grp] = st
        tr["apd_confidence"] = {g: {"confidence": round(c, 3)} for g, c in vd["conf"].items()}
        tr["apd_bbox"] = vd["bbox"]
