# Mask threshold on production's annotated validation set

348 frames never seen by the mask trunk, neither the frame nor its video. 845 annotated faces: **387 mask on**, 458 no mask. Pose (yolo11n, 0.40) located the head of 660 of them (305 mask on, 355 no mask); the rest are counted, not scored.

Compare the COCO human labels, which have 6 people with a mask on.

## Calibration: what the trunk's probability means

| p(mask) | faces | truly mask on | hard call at 0.5 right |
|---|---|---|---|
| 0.00-0.20 | 206 | 4% | 96% |
| 0.20-0.35 | 61 | 33% | 67% |
| 0.35-0.50 | 49 | 41% | 59% |
| 0.50-0.65 | 70 | 49% | 49% |
| 0.65-0.80 | 64 | 64% | 64% |
| 0.80-0.90 | 53 | 77% | 77% |
| 0.90-1.00 | 157 | 89% | 89% |

## Threshold sweep ('ok' only when p(mask) >= threshold; violation is positive)

| threshold | precision | recall | MCC | mask-on people wrongly flagged | violations missed |
|---|---|---|---|---|---|
| 0.30 | 91.0% | 65.4% | 0.592 | 23 of 305 | 123 of 355 |
| 0.40 | 87.6% | 69.9% | 0.588 | 35 of 305 | 107 of 355 |
| 0.50 | 84.5% | 75.2% | 0.590 | 49 of 305 | 88 of 355 |
| 0.60 | 80.9% | 83.7% | 0.609 | 70 of 305 | 58 of 355 |
| 0.70 | 76.2% | 86.5% | 0.562 | 96 of 305 | 48 of 355 |
| 0.80 | 72.4% | 91.8% | 0.548 | 124 of 305 | 29 of 355 |
| 0.85 | 70.0% | 92.7% | 0.511 | 141 of 305 | 26 of 355 |
| 0.90 | 67.2% | 95.2% | 0.481 | 165 of 305 | 17 of 355 |
| 0.95 | 62.9% | 97.5% | 0.409 | 204 of 305 | 9 of 355 |

Best MCC here: threshold 0.60 (0.609).

## Folders, at the deployed threshold 0.6

Every annotated face is saved into one folder; the box is the annotator's face box (green = mask on, red = no mask).

| folder | faces |
|---|---|
| `mask_sdk/correct_violation/` | 297 |
| `mask_sdk/correct_ok/` | 235 |
| `mask_sdk/not_located/` | 185 |
| `mask_sdk/WRONG_said_violation_truth_ok/` | 70 |
| `mask_sdk/WRONG_said_ok_truth_violation/` | 58 |

Faces under 30 px: 302 of 660 judged.

## What the wrong folders show

Contact sheets of 12 random faces each: `_sheet_WRONG_said_violation_truth_ok.jpg`,
`_sheet_WRONG_said_ok_truth_violation.jpg`, `_sheet_correct_ok.jpg`.

### Mask on, called a violation (`WRONG_said_violation_truth_ok/`)

The masks are visible, so the trunk is not missing something invisible. Three kinds:

- **Tiny faces.** Most are 11-23 px across, where a mask is a few pixels of
  white or green.
- **Masks that look like skin or shadow.** Black masks and white masks against
  pale backgrounds score low.
- **Faces turned down or away**, so only the edge of the mask shows.

A few clear, larger faces (33-34 px) with a white surgical mask still score
0.29-0.40, which is a plain model error rather than an input problem.

### No mask, called mask on (`WRONG_said_ok_truth_violation/`)

These are confident, 0.62-0.98, and often large, clear faces (42-79 px). Two
things recur:

- **Glasses.** About 5 of the 12 wear them. A dark or reflective band across the
  face seems to read as a mask. The earlier COCO error (a clear face with
  glasses called ok at 0.76) fits the same pattern.
- **Another person's mask inside the head crop.** In crowded shots the head
  region reaches a neighbour who is wearing one.

Faces seen from behind or looking down also default to "ok", because nothing
below the nose is visible.

### Why an obvious no-mask face gets "ok": the crop, not the face

Case study, `_case_glasses_office.jpg`: a clear, frontal, unmasked face 63 px
across, called ok at 0.66. What the mask trunk actually receives:

| input | p(mask) | call at 0.6 |
|---|---|---|
| deployed head crop | 0.66 | ok (wrong) |
| same, glasses painted out | 0.79 | ok (wrong) |
| crop moved down 30% | 0.34 | violation |
| annotator's tight face box | **0.05** | violation |

Glasses are not the trigger here: removing them made it worse. The crop is.
The head region is **1.9x the shoulder width** (`head_gain`, tuned for caps,
which need room above the head), and `_crop` then pads it by another 35% on
every side. For this person that is a ~330 px square around a 56 px head: the
face is about a seventh of the 128 px input and the mouth a few pixels. The
image is mostly whiteboard, air conditioner, shirt pattern and production's red
chevron.

Tightening the crop at inference does **not** fix it in general, because the
trunk was trained on these loose crops and has adapted to them. Across all 660
faces, best threshold for each crop:

| mask crop | MCC |
|---|---|
| deployed head ROI | **0.609** |
| nose-centred, 0.70x its width | 0.541 |
| nose-centred, 0.50x | 0.433 |
| nose-centred, 0.35x | 0.337 |
| nose-centred, 0.25x | 0.293 |

So the model makes its mask call on weak evidence it was trained to rely on.
Source domain is a minor factor: test clips and screenshots err 28% of the time
against 19% for CCTV frames, but they are 25 faces, and 53 of the 58
"no mask called ok" errors are ordinary CCTV frames.

### What would fix it

Neither error direction is a threshold problem, so moving the cut only trades
one for the other. The lever is retraining the **mask trunk on its own,
face-centred crop**, used identically in training and on the device, instead of
sharing head_cap's cap-sized region. The annotator face boxes in this set show
the idea works for a single face (0.66 -> 0.05). Add the two WRONG folders as
material: unmasked faces with glasses or in profile as hard negatives, small
masked faces as positives.

