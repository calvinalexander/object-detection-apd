# Mask on people facing away

`_unclear_but_judged_today.jpg`: people a human marked "mouth not visible" who
still received a mask verdict before this change. Most are backs of heads, the
SPPG logo on the shirt.

A mask can only be judged when the mouth is visible (the schema says so), but the
mask trunk ran on every head that passed the head gate. The pose keypoints
separate the two cases well:

| | nose confidence (median) | eye confidence (median) | left shoulder on image right |
|---|---|---|---|
| mouth not visible (COCO "unclear", 113) | 0.30 | 0.18 | 37% |
| judgeable faces (COCO, 19) | 0.93 | 0.85 | 89% |
| annotated faces, held-out videos (323) | - | - | 81% |

Facing the camera, a person's left shoulder and ear appear on the image's right;
turned away, the order flips. Gate now on the device: nose confidence >= 0.5, and
neither the shoulders nor the ears in turned-away order
(`scripts/104_mask_facing_gate.py`).

Through the deployed code path:

| | gate off | gate on |
|---|---|---|
| mask verdicts on the 113 mouth-not-visible people | 103 (91%) | **31 (27%)** |
| ... of which violation alerts | 60 | **22** |
| annotated test faces still judged | 294 of 323 | 238 of 323 |
| operational MCC on those test faces | 0.716 | 0.592 |
| COCO mask MCC | 0.300 | 0.271 |
| COCO head_cap MCC | 0.373 | 0.373 |

The cost is real: the annotators also labelled some faces the gate refuses, so
MCC on labelled faces goes down (abstaining counts as a missed alert). The gain
is 38 fewer violation alerts on people whose mouth nobody could see, which no
labelled-face metric can show. head_cap is untouched: a cap is visible from
behind.

Tunable on the device: `APD_OURS_MASK_MIN_NOSE` (0.5) and
`APD_OURS_MASK_REQUIRE_FACING` (1). Setting them to 0 and 0 restores the old
behaviour.
