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
