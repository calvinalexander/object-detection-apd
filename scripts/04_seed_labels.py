"""Convert the 592 human-drawn boxes into labels under the confirmed rules.

Two things follow from the rules confirmed 2026-09-08.

`gloves` is evaluated per HAND, not per person. The rule is "both hands gloved",
and the clean way to implement a conjunction is to learn the atom and compose it:
a classifier answers "is this hand gloved", and person-level compliance is an AND
over that person's hands. This also doubles the gloves training data and removes a
defect in the previous derivation, which stamped one person-level label onto both
hand crops and so mislabelled every one-gloved worker's second hand.

`mask` derivation is unchanged here because the seed boxes are the annotator's
own judgement and are taken as authoritative -- but note the stricter rule
(surgical only, must cover the nose) applies to everything labelled from now on.
"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apd.schema import Attribute, Value, LEGACY_COCO_CATEGORIES, assert_contract
from apd.pose import frontality

FRONTALITY_MIN = 0.5

assert_contract()
EXPORT = Path("data/sppg/.exports/coco-1778210446.4251382.json")
OUT = Path("artifacts/seed_labels.jsonl")

coco = json.loads(EXPORT.read_text())
frame_of = {i["id"]: i["file_name"] for i in coco["images"]}
boxes = collections.defaultdict(lambda: collections.defaultdict(list))
for a in coco["annotations"]:
    x, y, w, h = a["bbox"]
    boxes[frame_of[a["image_id"]]][LEGACY_COCO_CATEGORIES[a["category_id"]]].append(
        (x + w / 2, y + h / 2))
annotated = set(boxes)

def inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]

counts = collections.Counter()
hand_counts = collections.Counter()
rows = []
for line in Path("artifacts/persons.jsonl").read_text().splitlines():
    p = json.loads(line)
    if p["frame"] not in annotated:
        continue
    gt = boxes[p["frame"]]
    head, hands = p["head_roi"], p["hand_rois"]
    labels, per_hand = {}, []

    # head_cap -- a cap reads from any angle, so only localisation is required
    if head is None or p["head_collision"]:
        labels[Attribute.HEAD_CAP.value] = Value.UNCLEAR.value
    else:
        labels[Attribute.HEAD_CAP.value] = (
            Value.PRESENT if any(inside(c, head) for c in gt.get("head_cap", []))
            else Value.ABSENT).value

    # mask -- needs the mouth region, so the person must face the camera
    # Mask needs the mouth region, which keypoint confidence does not establish --
    # a pose model reports a nose it cannot see. Geometry does establish it.
    if head is None or p["head_collision"] or frontality(p["kp"]) < FRONTALITY_MIN:
        labels[Attribute.MASK.value] = Value.UNCLEAR.value
    else:
        labels[Attribute.MASK.value] = (
            Value.PRESENT if any(inside(c, head) for c in gt.get("mask", []))
            else Value.ABSENT).value

    # gloves -- per hand, then AND. Fewer than two visible hands cannot establish
    # "both", so the person-level answer is UNCLEAR even if the one hand we see
    # is gloved.
    for h_i, box in enumerate(hands):
        v = (Value.PRESENT if any(inside(c, box) for c in gt.get("gloves", []))
             else Value.ABSENT).value
        per_hand.append({"hand": h_i, "value": v})
        hand_counts[v] += 1
    if len(hands) < 2:
        labels[Attribute.GLOVES.value] = Value.UNCLEAR.value
    else:
        labels[Attribute.GLOVES.value] = (
            Value.PRESENT if all(h["value"] == Value.PRESENT.value for h in per_hand)
            else Value.ABSENT).value

    for a_, v in labels.items():
        counts[(a_, v)] += 1
    rows.append({"person_id": p["person_id"], "frame": p["frame"],
                 "scene_id": p["scene_id"], "labels": labels,
                 "hands": per_hand, "source": "seed_box"})

OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"{len(rows)} person instances across {len(annotated)} annotated frames\n")
print(f"{'attribute':<11}{'present':>9}{'absent':>9}{'unclear':>9}{'usable':>9}")
for attr in Attribute:
    p_, a_, u_ = (counts[(attr.value, v.value)] for v in
                  (Value.PRESENT, Value.ABSENT, Value.UNCLEAR))
    print(f"{attr.value:<11}{p_:>9}{a_:>9}{u_:>9}{100*(p_+a_)/(p_+a_+u_):>8.0f}%")
print(f"\nper-hand gloves labels: {hand_counts['present']} gloved / "
      f"{hand_counts['absent']} bare  ({sum(hand_counts.values())} hands)")
print(f"wrote {OUT}")
