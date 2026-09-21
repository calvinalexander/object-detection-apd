"""Pose-derived anatomical ROIs.

Pose replaces a plain person detector: one forward pass returns boxes and 17
keypoints, and the keypoints place the crops. Cropping to the part is what makes
the small classes learnable -- a mask is 7x5 px in a full frame resized to 640,
and 28x24 px in a head crop, about 20x the pixel evidence.

Two things the keypoints give us for free beyond placement:

* an abstain rule -- no visible face keypoints means head attributes are
  `unclear` rather than guessed;
* a collision test -- if another person's head falls inside this ROI, the crop is
  ambiguous about whose head it shows. 31% of head ROIs are affected, and they
  measured 60% labelling accuracy against 88% for clean ROIs, so they are routed
  to human review instead of being auto-labelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import (ARM_KEYPOINTS, FACE_KEYPOINTS, FRONTAL_KEYPOINTS, GEOMETRY,
                     L_EAR, L_EYE, NOSE, R_EAR, R_EYE, SHOULDER_KEYPOINTS, RoiGeometry)

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class PersonRois:
    person_box: Box
    head: Box | None
    hands: tuple[Box, ...]
    #: Any of nose/eyes/ears visible -- enough to place a head ROI, and enough to
    #: judge a cap, which is visible from behind.
    head_localizable: bool
    #: Nose, or both eyes, visible -- the person is facing the camera. Required to
    #: judge a mask; ears alone do not qualify, which is why this is separate.
    frontal_face: bool
    #: Another person's head centre falls inside this person's head ROI.
    head_collision: bool
    #: Face region for the mask trunk; None when no face keypoint is visible.
    face: Box | None = None


def _visible(keypoints: np.ndarray, indices, geom: RoiGeometry) -> list[np.ndarray]:
    return [keypoints[i] for i in indices if keypoints[i][2] >= geom.kp_visible]


def is_frontal(keypoints: np.ndarray, geom: RoiGeometry = GEOMETRY) -> bool:
    """Is the person facing the camera enough for the mouth region to be visible?

    Ears are visible from directly behind, so they cannot gate a mask judgement --
    only the nose, or both eyes, establishes that we are looking at a face.
    """
    if keypoints[NOSE][2] >= geom.kp_visible:
        return True
    return bool(keypoints[L_EYE][2] >= geom.kp_visible and keypoints[R_EYE][2] >= geom.kp_visible)


def head_centre(keypoints: np.ndarray, geom: RoiGeometry = GEOMETRY):
    """Centroid of the visible face keypoints, or None when no head is locatable."""
    face = _visible(keypoints, FACE_KEYPOINTS, geom)
    if not face:
        return None
    return (sum(p[0] for p in face) / len(face), sum(p[1] for p in face) / len(face))


def head_roi(keypoints: np.ndarray, person: Box, geom: RoiGeometry = GEOMETRY) -> Box | None:
    """Head ROI, scaled by shoulder width and biased upward to include a cap."""
    shoulders = _visible(keypoints, SHOULDER_KEYPOINTS, geom)
    centre = head_centre(keypoints, geom)

    if centre is not None:
        cx, cy = centre
    elif len(shoulders) == 2:
        # Turned away: infer the head above the shoulder midpoint.
        cx = (shoulders[0][0] + shoulders[1][0]) / 2
        cy = (shoulders[0][1] + shoulders[1][1]) / 2 - 0.35 * abs(shoulders[0][0] - shoulders[1][0])
    else:
        return None

    person_height = person[3] - person[1]
    scale = abs(shoulders[0][0] - shoulders[1][0]) if len(shoulders) == 2 else 0.0
    scale = max(scale, 0.20 * person_height)

    width = geom.head_gain * scale
    height = geom.head_gain * 1.08 * scale
    return (cx - width / 2, cy - height * geom.head_up,
            cx + width / 2, cy + height * (1 - geom.head_up))


def hand_rois(keypoints: np.ndarray, person: Box,
              geom: RoiGeometry = GEOMETRY) -> tuple[Box, ...]:
    """One ROI per visible wrist, sized from that arm's forearm length."""
    person_height = person[3] - person[1]
    out: list[Box] = []
    for wrist_i, elbow_i in ARM_KEYPOINTS:
        wrist = keypoints[wrist_i]
        if wrist[2] < geom.kp_visible:
            continue  # no wrist, no hand crop -- and no wasted classifier call
        elbow = keypoints[elbow_i]
        if elbow[2] >= geom.kp_visible:
            forearm = float(np.hypot(wrist[0] - elbow[0], wrist[1] - elbow[1]))
        else:
            forearm = 0.16 * person_height
        size = max(geom.hand_gain * forearm, 0.07 * person_height)
        out.append((wrist[0] - size / 2, wrist[1] - size / 2,
                    wrist[0] + size / 2, wrist[1] + size / 2))
    return tuple(out)


def face_roi(keypoints: np.ndarray, person: Box, geom: RoiGeometry = GEOMETRY,
             gain: float | None = None, down: float | None = None) -> Box | None:
    """Square face region centred on the mouth, for the mask trunk.

    Centre: the nose when visible, else the eye midpoint, else the face-keypoint
    centroid. Scale: shoulder width when both shoulders show, else the ear span
    scaled up, else a fraction of the person box. It is the same function in
    training and on the device, which is the whole point.
    """
    vis = geom.kp_visible
    gain = geom.face_gain if gain is None else gain
    down = geom.face_down if down is None else down
    if keypoints[NOSE][2] >= vis:
        cx, cy = float(keypoints[NOSE][0]), float(keypoints[NOSE][1])
    elif keypoints[L_EYE][2] >= vis and keypoints[R_EYE][2] >= vis:
        cx = float(keypoints[L_EYE][0] + keypoints[R_EYE][0]) / 2
        cy = float(keypoints[L_EYE][1] + keypoints[R_EYE][1]) / 2
    else:
        c = head_centre(keypoints, geom)
        if c is None:
            return None
        cx, cy = c
    shoulders = _visible(keypoints, SHOULDER_KEYPOINTS, geom)
    ears = [keypoints[i] for i in (L_EAR, R_EAR) if keypoints[i][2] >= vis]
    pw, ph = person[2] - person[0], person[3] - person[1]
    if len(shoulders) == 2:
        scale = abs(float(shoulders[0][0] - shoulders[1][0]))
    elif len(ears) == 2:
        scale = 2.2 * abs(float(ears[0][0] - ears[1][0]))
    else:
        scale = 0.5 * pw
    side = max(gain * scale, 0.06 * ph, 8.0)
    cy += down * side
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def _contains(point, box: Box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def build_rois(boxes: np.ndarray, keypoints: np.ndarray,
               geom: RoiGeometry = GEOMETRY) -> list[PersonRois]:
    """Derive every person's ROIs for one frame, including the collision flag."""
    heads = [head_roi(kp, tuple(box), geom) for box, kp in zip(boxes, keypoints)]
    centres = [head_centre(kp, geom) for kp in keypoints]

    out: list[PersonRois] = []
    for i, (box, kp) in enumerate(zip(boxes, keypoints)):
        roi = heads[i]
        collision = False
        if roi is not None:
            collision = any(
                centres[j] is not None and _contains(centres[j], roi)
                for j in range(len(keypoints)) if j != i
            )
        out.append(PersonRois(
            person_box=tuple(box),
            head=roi,
            hands=hand_rois(kp, tuple(box), geom),
            head_localizable=centres[i] is not None,
            frontal_face=is_frontal(kp, geom),
            head_collision=collision,
            face=face_roi(kp, tuple(box), geom),
        ))
    return out


# ---------------------------------------------------------------------------
# Frontality
# ---------------------------------------------------------------------------

def frontality(kp, kp_visible: float = 0.35) -> float:
    """0..1 estimate that the face points at the camera.

    Keypoint confidence cannot do this job. A pose model reports where it thinks
    a nose is even when that nose is on the far side of the head, so a
    high-confidence nose is entirely compatible with a view of someone's back --
    which is how a sheet of back-turned workers and one empty floor tile came to
    pass a "frontal_face" filter.

    Geometry does distinguish the cases. On a face turned toward the camera both
    eyes are visible and separated, and the nose sits between them. In profile
    the eyes converge and the nose leaves that span; from behind the ears are
    what remain. Score combines those two observations.

    kp is [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder] as
    (x, y, conf).
    """
    nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh = kp[:7]
    vis = lambda p: p[2] >= kp_visible

    if not (vis(l_eye) and vis(r_eye)):
        return 0.0                      # both eyes are the minimum for a face-on view

    eye_span = abs(l_eye[0] - r_eye[0])
    # Scale reference: shoulder width if we have it, else the ear span, else the
    # eye span itself (which then yields a neutral ratio).
    if vis(l_sh) and vis(r_sh):
        scale = abs(l_sh[0] - r_sh[0])
    elif vis(l_ear) and vis(r_ear):
        scale = abs(l_ear[0] - r_ear[0]) * 2.2
    else:
        return 0.35                     # eyes only: possible, unconfirmed
    if scale <= 1:
        return 0.0

    # A face-on head shows an eye separation near a third of shoulder width;
    # in profile it collapses toward zero.
    ratio = eye_span / scale
    span_score = min(1.0, ratio / 0.28)

    # The nose should sit inside the eye span when the head faces the camera.
    if vis(nose):
        lo, hi = sorted((l_eye[0], r_eye[0]))
        if eye_span > 1:
            excess = max(0.0, max(lo - nose[0], nose[0] - hi)) / eye_span
            nose_score = max(0.0, 1.0 - excess)
        else:
            nose_score = 0.0
    else:
        nose_score = 0.3                # no nose is weak evidence against face-on

    return round(0.6 * span_score + 0.4 * nose_score, 3)


# ---------------------------------------------------------------------------
# Hand ROI quality
# ---------------------------------------------------------------------------

L_WRIST_I, R_WRIST_I = 9, 10

def hand_quality(kp, person: Box, roi: Box, hand_index: int,
                 frame_w: int, frame_h: int, geom: RoiGeometry = GEOMETRY) -> float:
    """0..1 confidence that this ROI actually contains a hand.

    `gloves` was the only attribute with no gate of its own, and it showed: the
    ROI is placed wherever a wrist keypoint lands, nothing checks a hand ended up
    inside it, and a crop of a crate or a table edge scores low and is reported
    ABSENT -- a violation. Held out, the classifier came in at AUC 0.63 [0.50-0.75],
    losing to a constant "bare" prediction.

    Three cheap signals, multiplied:
      * wrist keypoint confidence -- the ROI's whole basis
      * how much of the ROI lies inside the frame -- edge crops are mostly padding
      * ROI size relative to the person -- a wildly large or tiny box means the
        forearm estimate failed
    """
    wrists = [kp[i] for i in (L_WRIST_I, R_WRIST_I) if len(kp) > i and kp[i][2] >= geom.kp_visible]
    if hand_index >= len(wrists):
        return 0.0
    conf = float(wrists[hand_index][2])

    x1, y1, x2, y2 = roi
    inside_w = max(0.0, min(x2, frame_w) - max(x1, 0.0))
    inside_h = max(0.0, min(y2, frame_h) - max(y1, 0.0))
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    inside = (inside_w * inside_h) / area

    person_h = max(1e-6, person[3] - person[1])
    rel = (y2 - y1) / person_h
    plausible = 1.0 if 0.05 <= rel <= 0.55 else 0.0

    return round(conf * inside * plausible, 3)


# --------------------------------------------------------------------------
# ROI admissibility
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RoiCheck:
    """Whether an ROI may be classified, and why not when it may not."""

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:          # so callers can write `if check:`
        return self.ok


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap_frac(box, other) -> float:
    iw = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
    ih = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
    return (iw * ih) / max(1e-6, _area(box))


def roi_usable(box, frame_shape, *, part: str = "head", overlay=None,
               others=(), self_box=None, geom: RoiGeometry = GEOMETRY) -> RoiCheck:
    """Gate an ROI before it reaches a classifier.

    Four ways an ROI can be unanswerable, all observed in review:

      clipped    the head runs off the edge of the frame, so the cap would be
                 outside the picture whether or not it is worn
      covered    a dashboard caption is painted over it; the pixels underneath
                 are gone, and removing the text does not bring them back
      occluded   a nearer worker stands in front of it
      too small  a head a couple of dozen pixels across carries no cap texture

    Returns a reason rather than a bare False so callers can report WHY a person
    was skipped instead of silently dropping them.
    """
    if box is None:
        return RoiCheck(False, "no roi")

    h, w = frame_shape[:2]
    inside = _overlap_frac(box, (0, 0, w, h))
    if 1.0 - inside > geom.max_clip_frac:
        return RoiCheck(False, "clipped by frame edge")

    min_px = geom.head_min_px if part == "head" else geom.hand_min_px
    side = min(box[2] - box[0], box[3] - box[1])
    if side < min_px:
        return RoiCheck(False, f"too small ({side:.0f}px < {min_px}px)")

    if overlay is not None and overlay.covers(box):
        return RoiCheck(False, "overlaps a dashboard caption")

    # Only a NEARER person occludes. Depth is not available, so the lower box
    # bottom stands in for it: in a fixed overhead camera the person whose feet
    # are further down the frame is closer. Without that test a distant worker
    # behind the subject would be counted as covering them.
    for other in others:
        if self_box is not None and tuple(other) == tuple(self_box):
            continue
        if self_box is not None and other[3] <= self_box[3]:
            continue
        if _overlap_frac(box, other) > geom.max_occlusion_frac:
            return RoiCheck(False, "occluded by a nearer person")

    return RoiCheck(True)
