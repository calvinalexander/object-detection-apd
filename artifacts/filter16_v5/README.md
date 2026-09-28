# v5 as a second filter on production's 16 Sep alerts

Same events and method as `artifacts/filter16` (which stays as the v3 build):
production's own `person_bbox`, cropped from its own snapshot with +18% padding,
letterboxed to 320, judged by `runs/person_ppe_v5/efficientnet_b0_320`.

v5 = v4's audited data (125 label fixes from the per-person audit) plus training
inputs matched to what the device actually delivers: crop height drawn around
142 px (the production-box median) and downscaled with INTER_AREA so the network
upsamples it as it will at inference, JPEG quality 55-85, and 10-30% cut off one
edge on a quarter of samples (one production box in five clips the person).

Cuts fitted on the even hours of the manual curation, scored on the odd hours:

| item | known_above | present_below |
|---|---|---|
| head_cap | 0.95 | 0.05 |
| mask | 0.10 | 0.275 |
| gloves | 0.80 | 0.10 |

| folder | events |
|---|---|
| `flagged/` | **1,328** |
| `passed/` | **1,776** |

## How it compares

Held out (odd hours, never used to fit the cuts):

| model | catches | false alarms | MCC |
|---|---|---|---|
| v3 | 131/219 | 115/444 | 0.330 |
| v4 (audit only) | 98/219 | 92/444 | 0.250 |
| **v5** | **140/219** | **78/444** | **0.464** |

McNemar v5 vs v3 on that half: right-where-wrong 103 vs 57, p = 3.7e-4.

Whole curation (half of it was used for fitting, so this is optimistic):

| your verdict | v3 | v5 |
|---|---|---|
| flag -> flagged | 99/362 | 220/362 |
| flag -> passed | 263 | 142 |
| skip -> flagged | 49/852 | 157/852 |

`index.csv` carries a `curation` column with your original flag/skip, so the
disagreements can be filtered directly.

Built by `scripts/150_build_filter16_v5.py` from `artifacts/scores/v5_16sep_scores.json`.
