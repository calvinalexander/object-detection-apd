# Our classifier as a second filter on the OTHER site's production alerts

Source: `data/snapshot/23` with its own events database `data/snapshot/23/db` --
camera `masak` on device **ED-A69C5A-4396**, 2026-09-20 03:07 to 2026-09-23 00:24,
2,459 events (2,452 with a usable snapshot). This is a different kitchen from the
16/17 Sep material, which came from device ED-97CE82-3753.

Method identical to `artifacts/filter16`: production's own `person_bbox`, cropped
from production's own snapshot with +18% padding, letterboxed to 320, judged by
`deploy/apd_person/apd_ours_person_ppe_v3.onnx` with the cuts in its JSON
(mask `known_above` 0.98 plus the confident-absence override).

| folder | meaning | events |
|---|---|---|
| `flagged/` | we call at least one item a violation, whether or not it is the item the device flagged | **1,250** |
| `passed/` | we find nothing wrong on this person | **1,202** |

Production's alerts, re-judged:

| item | kept | dropped as "cannot see" | dropped as compliant |
|---|---|---|---|
| head_cap | 374 / 996 (38%) | 108 | **514** |
| mask | 265 / 346 (77%) | 79 | 2 |
| gloves | 135 / 1116 (12%) | 978 | 3 |
| **all** | **774 / 2458 (31%)** | 1165 | 519 |

Violations we add that production did not raise: mask 556, head_cap 225, gloves 36.

Two differences from the 16 Sep site, both unresolved:

- **head_cap**: there we kept 76% of production's alerts; here we call 514 of 996
  actively compliant. Either this kitchen wears a covering one of the two models
  counts and the other does not, or production over-fires here. A curation pass
  of ~100 of those 514 would settle it.
- **gloves**: 88% of production's gloves alerts become "cannot see". Gloves is
  also production's most frequent alert at this site (1,115 events).

`index.csv` carries both verdicts per item (`device_*` from the device's own
`apd_detail` table, `ours_*` from v3) and the six raw scores.

Built by `scripts/142_filter_snapshot23.py` (scoring) and
`scripts/143_filter23_folders.py` (folders).
