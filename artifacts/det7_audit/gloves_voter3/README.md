# Audit: voter3-sourced gloves / no_gloves boxes in det7 (2026-09-17)

`python scripts/111_audit_site_gloves.py --source voter3 [--score]`. voter3 is
production's apd_v2 + apd_v3 annotated val frames (scripts/33). In det7 these give
245 person crops from 92 frames, 367 gloves boxes. Reviewer: Claude, one pass.

| class | boxes | ok | wrong class | not a hand | unclear | wrong |
|---|---|---|---|---|---|---|
| gloves | 168 | 125 | 2 | 4 | 37 | 4% |
| no_gloves | 199 | 149 | 0 | 2 | 48 | 1% |
| all | 367 | 274 (75%) | 2 | 6 | 85 (23%) | 2% |

Wrong class: #35 and #283 (bare hands labelled gloves).
Not a hand: #80 (clothing), #83 (patterned shirt), #210 (a teal lid), #246/#247
(a round object), #251 (the dashboard's red chevron).

Labels are clean. The problem with this source is the content, not the labels:
- most frames are staged demos -- office staff, whiteboards, lanyards, people
  holding gloved hands up to the camera, walking on stairs;
- hands are large (median well above the 10-40 px seen in the recordings), sharp
  and front-facing; transparent gloves appear mostly this way;
- no_gloves is heavily office work: keyboards, mice, phones;
- 92 frames with overlapping person crops, so each hand appears 2-4 times.
What it teaches: a big, sharp, frontal hand. Not an overhead kitchen hand.
