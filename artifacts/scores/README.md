# Raw person-model scores on production's 16 Sep device events

One row per event: event id, timestamp, production's violation list, and the six
outputs (`p[0:3]` presence for head_cap/mask/gloves, `p[3:6]` judgeability).
Written by `scripts/147_score_events_v4.py --run runs/person_ppe_<v>/efficientnet_b0_320`.

| file | model | cuts fitted on the curation (even hours), scored on odd hours |
|---|---|---|
| v3_16sep_scores.json | runs/person_ppe_v3 | 131/219 catches, 115/444 false alarms, MCC 0.330 |
| v4_16sep_scores.json | runs/person_ppe_v4 (audit) | 98/219, 92/444, MCC 0.250 |
| v5_16sep_scores.json | runs/person_ppe_v5 (audit + device-matched inputs) | **140/219, 78/444, MCC 0.464** |

McNemar v5 vs v3 on the held-out half: 103 / 57, p = 3.7e-4.

Folder outputs built from these scores:
  artifacts/filter16/     v3, 16 Sep   (flagged 2024 / passed 1080)  -- do not overwrite
  artifacts/filter23/     v3, other site 20-23 Sep
  artifacts/filter16_v5/  v5, 16 Sep   (to be built)
