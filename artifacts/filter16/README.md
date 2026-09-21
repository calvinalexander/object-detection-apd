# Our classifier as a second filter on production's 16 Sep captures

Every violation event the device wrote on 2026-09-16 (`data/prod_out`, camera
`masak`, 3,104 events), re-judged from the device's own snapshot: its person box
is cropped with the usual +18% pad, letterboxed to 320, and passed through
`deploy/apd_person/apd_ours_person_ppe_v3.onnx` with the camera-calibrated cuts
from its JSON (judge only when `p_known >= known_above`, then violation when
`p_present < present_below`). Mask's `known_above` was raised 0.896 -> **0.98**
on 21 Sep: on production's person boxes the old cut judged 65% of people, against
the ~32% the calibration aimed at. Nothing else changed, and no model other than
production's detector and v3 is involved.

| folder | meaning | events |
|---|---|---|
| `flagged/` | we call at least one item a violation, whether or not it is the item the device flagged | **1,875** |
| `passed/` | we find nothing wrong on this person | **1,229** |

The split is on our full verdict, not only on the device's items: every person is
judged on all three. Of the 1,579 events where we clear everything the device
flagged, 268 still carry a violation we found ourselves (head_cap 108, mask 125,
gloves 26, and 9 with two items), so 1,229 people come out clean.
The filename carries our items, e.g. `..._cap-mask_f7794264.jpg`, or `_clear_`.

Files from the 44 clips that were replayed carry a `CLIP_` prefix
(152 flagged, 56 passed of 208). `index.csv` has one row per event with both
verdicts and the six raw scores.

Per flagged item (an event can carry several):

| item | kept | dropped as "cannot see" | dropped as compliant |
|---|---|---|---|
| head_cap | 1324 / 1734 (76%) | 87 | 323 |
| mask | 285 / 753 (38%) | 443 | 25 |
| gloves | 187 / 1312 (14%) | 1092 | 33 |
| **all** | **1796 / 3799 (47%)** | 1622 | 381 |

At the old mask cut of 0.896 the mask row read 540 / 753 (72%), 174 "cannot see",
39 compliant, and the totals 2051 / 3799 (54%).

Only the replayed clips (208 events, 309 items): head_cap 115/138, mask 18/79,
gloves 22/92, all 155/309.

Violations we add that the device missed: head_cap 155, mask 318 (816 before the
mask cut was raised), gloves 158.

There is no ground truth on this footage, so these folders say what the filter
removes, not what it was right to remove. On the labelled COCO people the same
filter dropped 11 genuine false alerts but also 17 real violations, 12 of them
gloves.

Built by `scripts/134_filter_prod_alerts.py` (scoring) and `scripts/135_filter16_folders.py` (these folders).
