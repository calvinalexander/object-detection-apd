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
