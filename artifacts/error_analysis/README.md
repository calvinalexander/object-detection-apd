# Error analysis -- deployed configuration on human labels

yolo11n-pose at 0.40. Packed trunks for head_cap and mask, 'ok' when p(present) >= 0.5 for head_cap and 0.6 for mask. Crop detector for gloves at 0.25.
Every labelled person is sorted into one outcome; a crop of each is in the folder of the same name.

## head_cap -- 202 labelled people

| outcome | mask thresh 0.5 | now (0.5) | median person height |
|---|---|---|---|
| `head_cap/not_located/` | 104 | 104 | 136 px |
| `head_cap/correct_ok/` | 42 | 42 | 370 px |
| `head_cap/correct_violation/` | 34 | 34 | 408 px |
| `head_cap/WRONG_said_ok_truth_violation/` | 14 | 14 | 287 px |
| `head_cap/WRONG_said_violation_truth_ok/` | 7 | 7 | 419 px |
| `head_cap/abstained/` | 1 | 1 | 166 px |

## mask -- 44 labelled people

| outcome | mask thresh 0.5 | now (0.6) | median person height |
|---|---|---|---|
| `mask/not_located/` | 25 | 25 | 180 px |
| `mask/correct_violation/` | 13 | 14 | 417 px |
| `mask/WRONG_said_ok_truth_violation/` | 4 | 3 | 407 px |
| `mask/correct_ok/` | 2 | 2 | 212 px |

## gloves -- 122 labelled people

| outcome | count | share | median person height |
|---|---|---|---|
| `gloves/abstained/` | 62 | 51% | 363 px |
| `gloves/not_located/` | 49 | 40% | 209 px |
| `gloves/correct_violation/` | 8 | 7% | 320 px |
| `gloves/correct_ok/` | 2 | 2% | 462 px |
| `gloves/WRONG_said_ok_truth_violation/` | 1 | 1% | 484 px |

## Reading it

### mask: the threshold, and why these 44 people cannot settle it

The COCO labels hold only **6 people with a mask on** (38 without), so the
"correct ok" row can never be more than a handful. An earlier pass moved the
mask threshold to 0.80 on the strength of those 44 people; that did not survive
a proper test.

`mask_sdk/` redoes it on production's annotated validation set, with every
face saved into outcome folders like the ones here (**235 in
`mask_sdk/correct_ok/`**, against 2 in `mask/correct_ok/`): 348
frames the mask trunk never saw (neither the frame nor its video), 660 faces
located by pose, **305 of them wearing a mask**.

| mask threshold | precision | recall | MCC | mask-wearers wrongly flagged |
|---|---|---|---|---|
| 0.50 | 84.5% | 75.2% | 0.590 | 49 of 305 |
| **0.60 (deployed now)** | 80.9% | 83.7% | **0.609** | 70 of 305 |
| 0.80 (the COCO-based choice) | 72.4% | 91.8% | 0.548 | 124 of 305 |

At 0.80 the device would accuse 41% of the people who are wearing a mask. The
trunk is also reasonably calibrated there: faces scored 0.65-0.80 are truly
masked 64% of the time, not the 0% the six COCO cases suggested. 0.60 is now the
default (`APD_OURS_MASK_THRESH`).

Two limits on that set: it uses production's annotation convention, and 302 of
the 660 judged faces are under 30 px. head_cap stays at 0.5
(`APD_OURS_HEADCAP_THRESH`); 84% of its predictions are below 0.2 or above 0.8,
so its errors are confident ones no threshold moves.

### gloves: silent, not wrong

Almost no wrong gloves calls. The problem is that most labelled people get no
call at all: some are never located by pose, and many are located but the crop
detector returns neither `gloves` nor `no_gloves` (`gloves/abstained/`). It finds
the person and the cap at 0.90+ while saying nothing about hands, sometimes
where gloves are plainly visible (`_sheet_gloves_abstained.jpg`).

| detector threshold | located people who would get a verdict |
|---|---|
| 0.25 (deployed) | 15% |
| 0.15 | 19% |
| 0.10 | 26% |
| 0.05 | 34% |

Lowering it buys coverage with weak boxes; their precision is not yet measured.
`APD_OURS_DET_CONF` sets it.

### ruled out for mask: the head crop cutting off the mouth

On the located people with a visible nose, the nose sits at 0.69 of the crop
height (median), never in the bottom quarter, so the mouth is inside the crop.

### the shared cause: people never located

About half of all labelled people are never matched to a pose detection
(`*/not_located/`). They are the small, distant ones: median height 136-209 px
against 320-480 px for people who were judged.
