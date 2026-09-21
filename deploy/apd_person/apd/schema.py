"""Label contract for APD (PPE) compliance detection.

This module is the single source of truth for classes, attribute values and the
ROI geometry. Training, labelling and serving all import from here; nothing else
declares a class list. An earlier iteration of this project kept two class lists
in two files and they disagreed on the index of `mask` vs `head_cap` -- that is
the failure this module exists to prevent.

Architecture is two-stage:

    pose  ->  anatomical ROI  ->  per-part classifier

Stage 1 detects people and returns COCO-17 keypoints in one pass. Keypoints place
a head ROI and up to two hand ROIs. A small classifier answers one question per
part. There is no PPE-to-person IoU assignment step and no negative object class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


# --------------------------------------------------------------------------
# Attributes
# --------------------------------------------------------------------------

class Attribute(str, Enum):
    """What we assess. One question per attribute, per person."""

    HEAD_CAP = "head_cap"
    MASK = "mask"
    GLOVES = "gloves"


class Value(str, Enum):
    """Every attribute takes one of these three.

    UNCLEAR is not a failure of the labeller -- it is the correct answer when the
    evidence is not there, and it is masked out of the loss rather than guessed.
    Kitchen workers face away from cameras roughly 43% of the time, so forcing a
    binary would manufacture that much label noise.
    """

    PRESENT = "present"
    ABSENT = "absent"
    UNCLEAR = "unclear"


#: The part each attribute is judged from.
ATTRIBUTE_PART: Final[dict[Attribute, str]] = {
    Attribute.HEAD_CAP: "head",
    Attribute.MASK: "head",
    Attribute.GLOVES: "hand",
}


# --------------------------------------------------------------------------
# The head_cap rule
# --------------------------------------------------------------------------

#: Verbatim rule. Put this in annotator instructions and in any model prompt.
#:
#: Measured cost of omitting it: a reader told only "is there a head covering"
#: scores 29% specificity, calling hijabs caps almost every time. The same reader
#: given the rule below scores 67% on a fresh sample. The gap is entirely this
#: paragraph, so it is not optional anywhere in the pipeline.
#:
#: Revised 2026-09-14. The earlier wording counted a peaked cap as PRESENT.
#: Review of the rendered verdicts found street hats and baseball caps being
#: passed as compliant, which they are not in a food-handling area -- only a
#: food-grade covering that contains hair qualifies. 345 crops were relocated
#: from `present` to `absent` on that basis (artifacts/sanitized_head_cap_hats.txt)
#: and this text is the contract those labels now follow. Any labelling run
#: against the older wording will disagree with the dataset.
HEAD_CAP_RULE: Final[str] = (
    "head_cap is PRESENT only when the person wears a food-grade head covering "
    "that contains the hair: a bouffant cap, a hairnet, a surgical cap, or a "
    "hygiene skullcap. "
    "A street hat is ABSENT, not present -- this includes a baseball cap, any "
    "peaked or brimmed cap, a beanie, and a bandana. Such items are not food-"
    "service PPE however neatly they are worn. "
    "A hijab, headscarf, hood or any other cloth covering worn WITHOUT a "
    "food-grade cap over it is ABSENT. A bare head is ABSENT. "
    "If a food-grade cap is worn over a hijab, that is PRESENT. "
    "If the top of the head cannot be seen, the answer is UNCLEAR, not ABSENT."
)

#: Mask. Confirmed 2026-09-08. Both clauses are strict and both matter: the
#: below-the-nose case is the single most common real-world posture, and counting
#: it as compliant would make the class nearly meaningless.
MASK_RULE: Final[str] = (
    "mask is PRESENT only when a surgical mask covers BOTH nose and mouth. "
    "A mask worn below the nose is ABSENT. A cloth or fabric mask is ABSENT. "
    "A mask pulled under the chin or hanging from one ear is ABSENT. "
    "If the mouth region cannot be seen, the answer is UNCLEAR, not ABSENT."
)

#: Gloves. Confirmed 2026-09-08. Gloves are required for the whole shift, so
#: there is no task-based exemption -- an ungloved worker is non-compliant
#: regardless of what they are doing.
GLOVES_RULE: Final[str] = (
    "gloves is PRESENT only when BOTH hands are gloved. One gloved hand is "
    "ABSENT. Gloves are required at all times, not only while handling food, so "
    "no activity exempts a worker. If fewer than two hands can be seen, the "
    "answer is UNCLEAR, not ABSENT."
)

#: Rule text keyed by attribute, so annotator instructions and model prompts draw
#: from one place rather than paraphrasing.
ATTRIBUTE_RULE: Final[dict[str, str]] = {}   # populated below, after Attribute

#: Sub-populations that head_cap accuracy must be reported against separately.
#: A single head_cap number looks healthy while hijab confusion runs at 30%,
#: because approved caps dominate the class.
HEAD_STATE_STRATA: Final[tuple[str, ...]] = ("bare", "cloth_covering", "approved_cap")


# --------------------------------------------------------------------------
# ROI geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RoiGeometry:
    """Anatomical ROI placement, fixed by sweep against the annotated frames.

    `head_gain` 1.9 / `head_up` 0.65 gave 91.9% head_cap and 93.5% mask
    containment; `hand_gain` 2.4 gave 93.5% for gloves. Larger gains raise
    containment but also raise the rate at which an ROI swallows a neighbour's
    head -- see `collision` handling in apd.pose.
    """

    head_gain: float = 1.9
    head_up: float = 0.65
    head_crop_px: int = 128

    hand_gain: float = 2.4
    hand_crop_px: int = 96

    #: Face region for the mask trunk, separate from the head region. The head
    #: region is sized for caps (1.9x shoulder width, lifted), which leaves the
    #: mouth a few pixels of a 128 px input. Side = face_gain x shoulder width,
    #: centred on the nose and pushed down by face_down x side toward the mouth.
    face_gain: float = 0.8
    face_down: float = 0.15
    face_crop_px: int = 128

    #: Keypoint confidence at or above which a joint counts as visible.
    kp_visible: float = 0.35

    #: Pose inference size. The frames are 704x576 and 2880x1620.
    pose_imgsz: int = 1920

    # ------------------------------------------------------------------
    # ROI admissibility. A classifier answers whatever it is handed, so the
    # decision not to ask has to be made here. Review of the rendered verdicts
    # found head_cap being decided on heads clipped by the frame edge, heads
    # behind a caption, heads hidden by a nearer worker, and heads a few dozen
    # pixels across. None of those are answerable, and a confident answer to an
    # unanswerable question is worse than an abstention.
    # ------------------------------------------------------------------

    #: Smallest usable ROI, shortest side in source pixels.
    head_min_px: int = 26
    hand_min_px: int = 18

    #: Largest share of an ROI that may fall outside the frame.
    max_clip_frac: float = 0.15

    #: Largest share of an ROI that may be covered by a NEARER person's box.
    max_occlusion_frac: float = 0.35
    pose_conf: float = 0.15


GEOMETRY: Final[RoiGeometry] = RoiGeometry()


# COCO-17 keypoint indices, named so the ROI code does not carry magic numbers.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

#: Enough to locate a head and judge a cap -- ears count, since a cap is visible
#: from behind.
FACE_KEYPOINTS: Final[tuple[int, ...]] = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
#: Enough to judge a mask -- the person must actually be facing the camera.
#: Gating `mask` on FACE_KEYPOINTS instead would count a view of the back of
#: someone's head as a readable face.
FRONTAL_KEYPOINTS: Final[tuple[int, ...]] = (NOSE, L_EYE, R_EYE)
SHOULDER_KEYPOINTS: Final[tuple[int, ...]] = (L_SHOULDER, R_SHOULDER)
#: (wrist, elbow) pairs -- the elbow sets the hand ROI scale via forearm length.
ARM_KEYPOINTS: Final[tuple[tuple[int, int], ...]] = ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW))


# --------------------------------------------------------------------------
# Legacy COCO export
# --------------------------------------------------------------------------

#: Category names in the coco-annotator exports the seed labels come from.
#: Read-only: used to convert boxes into per-part attribute values, never to
#: define a detector's output classes.
LEGACY_COCO_CATEGORIES: Final[dict[int, str]] = {
    3: "person",
    4: "gloves",
    5: "head_cap",
    6: "mask",
}


ATTRIBUTE_RULE.update({
    Attribute.HEAD_CAP.value: HEAD_CAP_RULE,
    Attribute.MASK.value: MASK_RULE,
    Attribute.GLOVES.value: GLOVES_RULE,
})


def assert_contract() -> None:
    """Fail loudly on a schema that has drifted. Call at process start."""
    assert set(ATTRIBUTE_PART) == set(Attribute), "ATTRIBUTE_PART misses an attribute"
    assert len(set(LEGACY_COCO_CATEGORIES.values())) == len(LEGACY_COCO_CATEGORIES), \
        "duplicate name in LEGACY_COCO_CATEGORIES"
    for attr in Attribute:
        assert ATTRIBUTE_PART[attr] in ("head", "hand"), f"unknown part for {attr}"
        assert ATTRIBUTE_RULE.get(attr.value), f"no stated rule for {attr}"
